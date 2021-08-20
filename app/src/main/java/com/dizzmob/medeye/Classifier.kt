package com.dizzmob.medeye

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList

class Classifier(val context: Context, assetManager: AssetManager) {

    private val modelPath = "model.tflite"
    private val labelPath = "labels.txt"

    private var interpreter: Interpreter
    private var labelList: List<String>

    private var INPUT_SIZE: Int = 224
    private val PIXEL_SIZE: Int = 3
    private val FLOAT_TYPE_SIZE: Int = 4

    private val IMAGE_MEAN = 0.0f
    private val IMAGE_STD = 1.0f

    private val MAX_RESULTS = 3
    private val THRESHOLD = 0.6f

    private var gpuDelegate: GpuDelegate? = null

    data class Recognition(var id: String = "", var label: String = "", var confidence: Float = 0F)  {

        override fun toString(): String {
            return "Label = $label, Confidence = $confidence)"
        }
    }

    init {
        try {
            val options = Interpreter.Options()
            //gpuDelegate = GpuDelegate()
            //options.addDelegate(gpuDelegate)
            interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
            interpreter.allocateTensors()

        } catch (e: Exception) {
            interpreter = Interpreter(loadModelFile(assetManager, modelPath))
            interpreter.allocateTensors()

            e.printStackTrace()
        }

        labelList = loadLabelList(assetManager, labelPath)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {

        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {

        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }
    }

    fun recognizeImage(bitmap: Bitmap): List<Recognition> {

        val scaledBitmap: Bitmap = getResizedBitmap(bitmap, INPUT_SIZE, INPUT_SIZE)
        //val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)

        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) { FloatArray(labelList.size) }

        val startTime = SystemClock.uptimeMillis()

        interpreter.run(byteBuffer, result)

        val endTime = SystemClock.uptimeMillis()

        val inferenceTime = endTime - startTime

        return getSortedResult(result)
    }

    private fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {

        val width = bm.width
        val height = bm.height

        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height

        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)

        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * FLOAT_TYPE_SIZE)

        byteBuffer.order(ByteOrder.nativeOrder())

        byteBuffer.rewind()

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val input = intValues[pixel++]

                byteBuffer.putFloat((((input.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((input.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((input and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
            }
        }

        return byteBuffer
    }

    private fun getSortedResult(labelProbArray: Array<FloatArray>): List<Recognition> {
        Log.d("Classifier", "List Size:(%d, %d, %d)".format(labelProbArray.size,labelProbArray[0].size,labelList.size))

        val pq = PriorityQueue(
            MAX_RESULTS,
            Comparator<Recognition> {
                    (_, _, confidence1), (_, _, confidence2)
                -> java.lang.Float.compare(confidence1, confidence2) * -1
            })

        for (i in labelList.indices) {
            val confidence = labelProbArray[0][i]

            if (confidence >= THRESHOLD) {
                pq.add(Recognition("" + i,
                    if (labelList.size > i) labelList[i] else "Unknown", confidence * 100.0f)
                )
            }
        }
        Log.d("Classifier", "pqsize:(%d)".format(pq.size))

        val recognitions = ArrayList<Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        return recognitions
    }

    fun close() {
        interpreter.close()
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
        }
    }
}