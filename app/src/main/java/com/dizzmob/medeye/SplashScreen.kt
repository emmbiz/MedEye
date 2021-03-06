package com.dizzmob.medeye

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity

class SplashScreen : AppCompatActivity() {

    private val SPLASH_DISPLAY_LENGTH: Long = 2700

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash_screen)

        // Display Splash Screen for about 2 seconds
        Handler(Looper.getMainLooper())
            .postDelayed({
                val mainIntent = Intent(this@SplashScreen, CameraActivity::class.java)
                this@SplashScreen.startActivity(mainIntent)
                finish()
            }, SPLASH_DISPLAY_LENGTH)
    }
}