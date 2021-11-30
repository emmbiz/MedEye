package com.dizzmob.medeye

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.Button
import android.widget.ImageView
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
import com.google.android.material.floatingactionbutton.FloatingActionButton
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraActivity : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null

    private var imageRotationDegrees: Int = 0

    private lateinit var bitmapBuffer: Bitmap

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var imagePredicted: ImageView
    private lateinit var textPrediction: Button
    private lateinit var floatingActionButton: FloatingActionButton

    private lateinit var classifier: Classifier

    private lateinit var outputDirectory: File

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss"

        private const val PERMISSIONS_REQUEST_CODE = 10
        private const val SELECT_IMAGE_REQUEST_CODE = 100

        private val DISPLAY_DIALOG_DELAY_LENGTH: Long = 3200

        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    private var pauseAnalysis = false
    private var captureButtonIsEnabled = false
    private var bitmapImage: Bitmap? = null
    private var resultConfidenceInfo: String? = null

    private val camera: Camera? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        viewFinder = findViewById<View>(R.id.viewFinder) as PreviewView
        captureButton = findViewById<View>(R.id.captureButton) as Button
        imagePredicted = findViewById<View>(R.id.imagePredicted) as ImageView
        textPrediction = findViewById<View>(R.id.textPrediction) as Button
        floatingActionButton = findViewById<View>(R.id.floatingActionButton) as FloatingActionButton

        initializeCamera()

        initializeClassifier()

        setListeners()

        outputDirectory = getOutputDirectory()
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

    private fun setListeners() {
        captureButton.setOnClickListener {
            if (captureButtonIsEnabled) {
                pauseAnalysis = true
                imagePredicted.rotation = 90F

                takePhoto()

            } else {
                val message = Toast.makeText(this@CameraActivity, getString(R.string.capture_disabled_message), Toast.LENGTH_LONG)
                message.setGravity(Gravity.CENTER, 0, 0)
                message.show()
            }
        }

        textPrediction.setOnClickListener {
            if (resultConfidenceInfo != null) {
                Toast.makeText(this@CameraActivity, resultConfidenceInfo, Toast.LENGTH_LONG).show()
            }
        }

        floatingActionButton.setOnClickListener {
            pauseAnalysis = false

            Thread.sleep(100)

            hideResultViews()
        }
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

            imageCapture = ImageCapture.Builder()
                //.setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetResolution(Size(1080, 1920))
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

                                if (result?.label == getString(R.string.no_xray_image_label) && result.confidence > 60.0) {
                                    captureButton.setBackgroundResource(R.drawable.button_disabled)

                                    captureButtonIsEnabled = false
                                } else {
                                    captureButton.setBackgroundResource(R.drawable.button_enabled)

                                    captureButtonIsEnabled = true
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

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                       this as LifecycleOwner, cameraSelector, preview, imageCapture, imageAnalyzer
                )

                setUpTapToFocus()

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

            imagePredicted.rotation = 0F

            analyzeImageForResult(imageUri!!)
        }
    }

    private fun analyzeImageForResult(imageUri: Uri) {
        try {
            bitmapImage = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)

            // Get maximum result by confidence
            val res = classifier.recognizeImage(bitmapImage!!)
            val result = res.maxBy { it.confidence }

            if (result == null) {
                textPrediction.text =  getString(R.string.unknown_image_label)
                textPrediction.setBackgroundResource(R.drawable.result_default)

                resultConfidenceInfo = null
            } else {
                textPrediction.text = result.label
                val confidence = "%.2f".format(result.confidence)
                resultConfidenceInfo = getString(R.string.result_confidence_info, confidence)

                if (result.label == getString(R.string.no_xray_image_label)) {
                    textPrediction.setBackgroundResource(R.drawable.result_default)
                } else if (result.label == getString(R.string.covid_label)) {
                    textPrediction.setBackgroundResource(R.drawable.result_covid_detected)

                    // Display Dialog after some seconds
                    Handler(Looper.getMainLooper())
                        .postDelayed({
                            showDialog()
                        }, DISPLAY_DIALOG_DELAY_LENGTH)

                } else {
                    textPrediction.setBackgroundResource(R.drawable.result_normal)
                }
            }

            showResultViews()

        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun showDialog() {
        val builder: MaterialAlertDialogBuilder = MaterialAlertDialogBuilder(this@CameraActivity)

        builder.setTitle(getString(R.string.result_dialog_title))
        builder.setMessage(getString(R.string.result_dialog_message))
        builder.setPositiveButton(getString(R.string.result_dialog_positive_button_text)) { _, _ ->

        }
        // Create and show alert dialog
        val dialog = builder.create()
        dialog.show()
    }

    private fun showResultViews() {
        title = getString(R.string.results_header_text)

        viewFinder.visibility = View.GONE
        captureButton.visibility = View.GONE

        imagePredicted.setImageBitmap(bitmapImage)
        imagePredicted.visibility = View.VISIBLE
        textPrediction.visibility = View.VISIBLE
        floatingActionButton.visibility = View.VISIBLE
    }

    private fun hideResultViews() {
        title = getString(R.string.app_title)

        imagePredicted.visibility = View.GONE
        textPrediction.visibility = View.GONE
        floatingActionButton.visibility = View.GONE
        imagePredicted.setImageBitmap(null)

        viewFinder.visibility = View.VISIBLE
        captureButton.visibility = View.VISIBLE
    }

    private fun hideStatusBar() {
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setUpTapToFocus() {
        val listener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val currentZoomRatio: Float = camera?.cameraInfo?.zoomState?.value?.zoomRatio ?: 1F
                val delta = detector.scaleFactor
                camera?.cameraControl?.setZoomRatio(currentZoomRatio * delta)
                return true
            }
        }

        val scaleGestureDetector = ScaleGestureDetector(this, listener)

        viewFinder.setOnTouchListener { _, event ->
            scaleGestureDetector.onTouchEvent(event)
            if (event.action == MotionEvent.ACTION_DOWN || event.action == MotionEvent.ACTION_UP) {
                val factory = viewFinder.meteringPointFactory
                val autoFocusPoint = factory.createPoint(event.x, event.y)
                val autoFocusAction = FocusMeteringAction.Builder(autoFocusPoint, FocusMeteringAction.FLAG_AF)
                        .setAutoCancelDuration(5, TimeUnit.SECONDS)
                        .build()
                camera?.cameraControl?.startFocusAndMetering(autoFocusAction)
            }
            true
        }
    }

    private fun setUpAutoFocus() {
        val factory: MeteringPointFactory = SurfaceOrientedMeteringPointFactory(viewFinder.width.toFloat(), viewFinder.height.toFloat())
        //val factory2 = SurfaceOrientedMeteringPointFactory(1f, 1f)

        val centerWidth = viewFinder.width.toFloat() / 2
        val centerHeight = viewFinder.height.toFloat() / 2
        val autoFocusPoint = factory.createPoint(centerWidth, centerHeight)
        //val autoFocusPoint2 = factory.createPoint(.5f, .5f)

        try {
            val autoFocusAction: FocusMeteringAction = FocusMeteringAction.Builder(autoFocusPoint, FocusMeteringAction.FLAG_AF)
                    .setAutoCancelDuration(5, TimeUnit.SECONDS)
                    .build()
            camera?.cameraControl?.startFocusAndMetering(autoFocusAction)
        } catch (exc: Exception) {
            Log.d("ERROR", "cannot access camera", exc)
        }
    }

    private fun getOutputDirectory(): File {

        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
            File(mFile, resources.getString(R.string.app_name)).apply {
                mkdirs()
            }
        }

        return if (mediaDir != null && mediaDir.exists()) mediaDir else filesDir
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
                outputDirectory,
                SimpleDateFormat(FILENAME_FORMAT, Locale.getDefault()).format(System.currentTimeMillis()) + ".jpg"
        )

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has been taken
        imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(this@CameraActivity),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        val imageUri = Uri.fromFile(photoFile)

                        analyzeImageForResult(imageUri)
                    }

                    override fun onError(exc: ImageCaptureException) {
                        pauseAnalysis = false

                        Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    }
                }
        )
    }

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

        return true
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