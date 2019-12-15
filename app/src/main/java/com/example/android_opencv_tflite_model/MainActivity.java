package com.example.android_opencv_tflite_model;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
//import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;
import android.graphics.Color;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;

import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.view.ViewGroup.LayoutParams;

import android.widget.SeekBar.OnSeekBarChangeListener;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    //view holder
    CameraBridgeViewBase cameraBridgeViewBase;

    //camera listener callback
    BaseLoaderCallback baseLoaderCallback;

    private static final String TAG = "Activity::TfLite";
    private static final int PREVIEW_SIZE = 120;

    private Mat mRgba;
    private Mat intermediate;
    private Mat input;
    private TextView tv;
    private Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tv = (TextView)findViewById(R.id.text1);

        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.cameraViewer);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        baseLoaderCallback = new BaseLoaderCallback(this) { //create camera listener callback
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.i(TAG, "loader interface success");

                        if(classifier!=null) {
                            classifier.classifyMat(input);
                        }
                        cameraBridgeViewBase.setCameraIndex(0);// 0-for reverse camera, 1-for frontal camera
                        cameraBridgeViewBase.enableFpsMeter();// Frame per seconds meter for check load over processor
                        cameraBridgeViewBase.enableView();
                        break;

                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };

    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        mRgba = new Mat();
        intermediate = new Mat();
        input = new Mat();

    }

    @Override
    public void onCameraViewStopped() {
        if(intermediate!=null)
            intermediate.release();
        if(input!=null)
            input.release();
        if(mRgba!=null)
            mRgba.release();

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba=inputFrame.rgba();
        Mat mGray = inputFrame.gray();

        int top = mRgba.rows()/2 - PREVIEW_SIZE;
        int left = mRgba.cols() / 2 - PREVIEW_SIZE;
        int height = PREVIEW_SIZE*2;
        int width = PREVIEW_SIZE*2;

        Mat sample;

        //draw ROI over camera frame
        Imgproc.rectangle(mRgba, new Point(mRgba.cols()/2 - PREVIEW_SIZE, mRgba.rows() / 2 - PREVIEW_SIZE), new Point(mRgba.cols() / 2 + PREVIEW_SIZE, mRgba.rows() / 2 + PREVIEW_SIZE), new Scalar(255,255,255),2);

        //crop ROI frame
        Mat temp = mGray.submat(top, top + height, left, left + width);

        //convert gray frame to binary using apadative thresold
        Imgproc.GaussianBlur(temp,temp, new Size(5,5),0);
        Imgproc.adaptiveThreshold(temp, intermediate, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C , Imgproc.THRESH_BINARY_INV, 7, 7);

        Imgproc.resize(intermediate, input, new org.opencv.core.Size(28,28));///CNN input

        //show preprocessed cropped region at top left of camera frame
        sample= mRgba.submat(top,   mRgba.rows()/2 + PREVIEW_SIZE, left, mRgba.cols() / 2 + PREVIEW_SIZE);

        //cover grayscale to BGRA
        Imgproc.cvtColor(intermediate, sample, Imgproc.COLOR_GRAY2BGRA, 4);

        // run classifier
        classifier.classifyMat(input);
        Imgproc.putText( mRgba, "Inference:"+classifier.getDigit(), new Point( mRgba.rows()/2, mRgba.cols()/8 ), 0, 1, new Scalar(255, 255, 255, 255), 2);

        temp.release();
        sample.release();

        return mRgba;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }

        //stop classifier
        if(classifier!=null) {
            classifier.close();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There is a problem", Toast.LENGTH_SHORT).show();
        } else {
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }

        try {
            classifier = new Classifier(MainActivity.this);

        } catch (IOException e) {
            Log.e(TAG, "error", e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

}
