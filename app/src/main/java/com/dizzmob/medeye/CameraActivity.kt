package com.dizzmob.medeye

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.provider.MediaStore
import android.util.Log
import android.view.*
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.dizzmob.medeye.utils.YuvToRgbConverter
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private var imageRotationDegrees: Int = 0

    private lateinit var bitmapBuffer: Bitmap

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var imagePredicted: ImageView
    private lateinit var textPrediction: TextView

    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private lateinit var classifier: Classifier

    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val PERMISSIONS_REQUEST_CODE = 10
        private const val SELECT_IMAGE_REQUEST_CODE = 100

        private val DISPLAY_DIALOG_DELAY_LENGTH = 2700

        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    private var pauseAnalysis = false
    private var bitmapImage: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        viewFinder = findViewById<View>(R.id.viewFinder) as PreviewView
        captureButton = findViewById<View>(R.id.captureButton) as Button
        imagePredicted = findViewById<View>(R.id.imagePredicted) as ImageView
        textPrediction = findViewById<View>(R.id.textPrediction) as TextView

        initializeCamera()

        initializeClassifier()
    }

    private fun initializeCamera() {
        if (allPermissionsGranted()) startCamera()
        else {
            ActivityCompat.requestPermissions(this@CameraActivity, REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST_CODE)
        }
    }

    private fun allPermissionsGranted() =
            REQUIRED_PERMISSIONS.all {
                ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
            }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            PERMISSIONS_REQUEST_CODE -> {
                if (allPermissionsGranted()) startCamera()
                else {
                    Toast.makeText(this@CameraActivity, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
    }

    private fun initializeClassifier() {
        classifier = Classifier(this@CameraActivity, assets)
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this@CameraActivity)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    //.setTargetRotation(viewFinder.display.rotation)
                    .build()

            val converter = YuvToRgbConverter(this@CameraActivity)

            // Set up the image analysis use case which will process frames in real time
            val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    //.setTargetRotation(viewFinder.display.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { imgAnalyzer: ImageAnalysis ->
                        // Set up the image analyzer
                        imgAnalyzer.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { image ->

                            if (!::bitmapBuffer.isInitialized) {
                                // The image rotation and RGB image buffer are initialized just once
                                imageRotationDegrees = image.imageInfo.rotationDegrees

                                bitmapBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                            }

                            // Early exit: image analysis is in paused state
                            if (pauseAnalysis) {
                                image.close()
                                return@Analyzer
                            }

                            // Convert the image to RGB and place it in our shared buffer
                            image.use { converter.yuvToRgb(image.image!!, bitmapBuffer) }

                            // Get maximum result by confidence
                            val res = classifier.recognizeImage(bitmapBuffer)
                            val result = res.maxBy { it.confidence }

                            runOnUiThread {

                                if (result == null) {
                                    captureButton.visibility = View.INVISIBLE
                                } else {
                                    captureButton.text = result.label
                                    captureButton.visibility = View.VISIBLE

                                    if (result.label == getString(R.string.no_xray_image_label)) {
                                        captureButton.setBackgroundResource(R.drawable.result_default)
                                    } else if (result.label == getString(R.string.covid_label)) {
                                        captureButton.setBackgroundResource(R.drawable.result_covid_detected)
                                    } else {
                                        captureButton.setBackgroundResource(R.drawable.result_normal_detected)
                                    }
                                }
                            }
                        })
                    }

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                       this as LifecycleOwner, cameraSelector, preview, imageAnalyzer
                )

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(viewFinder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this@CameraActivity))
    }

    private fun openGallery() {
        val gallery = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI)
        startActivityForResult(gallery, SELECT_IMAGE_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        pauseAnalysis = false

        if (resultCode == Activity.RESULT_OK && requestCode == SELECT_IMAGE_REQUEST_CODE && data != null && data.data != null) {
            val imageUri: Uri? = data.data

            try {
                bitmapImage = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)

                // Get maximum result by confidence
                val res = classifier.recognizeImage(bitmapImage!!)
                val result = res.maxBy { it.confidence }

                if (result == null) {
                    captureButton.visibility = View.INVISIBLE
                } else {
                    captureButton.text = result.label
                    captureButton.visibility = View.VISIBLE

                    if (result.label == getString(R.string.no_xray_image_label)) {
                        captureButton.setBackgroundResource(R.drawable.result_default)
                    } else if (result.label == getString(R.string.covid_label)) {
                        captureButton.setBackgroundResource(R.drawable.result_covid_detected)

                        // Display Dialog after some seconds
                        Handler()
                                .postDelayed({
                                    showDialog()
                                }, DISPLAY_DIALOG_DELAY_LENGTH.toLong())
                    } else {
                        captureButton.setBackgroundResource(R.drawable.result_normal_detected)
                    }

                    showResultViews()
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    private fun showDialog() {
        //val builder: AlertDialog.Builder = AlertDialog.Builder(this)
        val builder: MaterialAlertDialogBuilder = MaterialAlertDialogBuilder(this@CameraActivity)

        builder.setTitle(getString(R.string.result_dialog_title))
        builder.setMessage(getString(R.string.result_dialog_message))
        builder.setPositiveButton(getString(R.string.result_dialog_positive_button_text)) { dialog, id ->
            Toast.makeText(this@CameraActivity, getString(R.string.result_dialog_feedback), Toast.LENGTH_SHORT).show()
        }

        // create and show alert dialog
        val dialog = builder.create()
        dialog.show()
    }

    private fun showResultViews() {
        pauseAnalysis = true

        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            supportActionBar!!.setHomeButtonEnabled(true)
            supportActionBar!!.setHomeAsUpIndicator(R.drawable.ic_baseline_arrow_back_24)
            supportActionBar!!.setDisplayShowHomeEnabled(true)
        }
        title = getString(R.string.results_header_text)

        viewFinder.visibility = View.GONE

        imagePredicted.setImageBitmap(bitmapImage)
        imagePredicted.visibility = View.VISIBLE
        textPrediction.visibility = View.VISIBLE
    }

    private fun hideResultViews() {
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(false)
            supportActionBar!!.setHomeButtonEnabled(false)
            supportActionBar!!.setHomeAsUpIndicator(R.drawable.ic_baseline_arrow_back_24)
            supportActionBar!!.setDisplayShowHomeEnabled(false)
        }
        title = getString(R.string.app_title)

        imagePredicted.visibility = View.GONE
        textPrediction.visibility = View.GONE
        captureButton.text = getString(R.string.default_button_text)
        captureButton.setBackgroundResource(R.drawable.result_default)

        viewFinder.visibility = View.VISIBLE
    }

    /** create options menu and perform action based on menu item selected **/
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        super.onCreateOptionsMenu(menu)
        menuInflater.inflate(R.menu.menu_camera, menu)

        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.uploadImageItem) {
            pauseAnalysis = true

            openGallery()
        }

        if (item.itemId == android.R.id.home) {
            pauseAnalysis = false

            hideResultViews()
        }

        return true
    }

    override fun onBackPressed() {
        pauseAnalysis = false

        hideResultViews()
    }

    override fun onResume() {
        super.onResume()
        initializeCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
        cameraExecutor.shutdown()
    }
}