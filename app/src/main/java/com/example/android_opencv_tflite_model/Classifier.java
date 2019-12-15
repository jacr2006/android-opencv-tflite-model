package com.example.android_opencv_tflite_model;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;
import android.os.FileUtils;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.FileInputStream;
import java.io.IOException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/**
 * Created by rohithkvsp on 4/22/18.
 * Modified by Jose Carrasquel
 */

public class Classifier {

    private static final String TAG = "Classifier::TfLite";

    private static final int INPUT_BATCH_SIZE = 1;
    private static final int INPUT_PIXEL_SIZE =1;
    private static final int INPUT_HEIGHT =28;
    private static final int INPUT_WIDTH = 28;
    private static final int BYTES =4;

    protected Interpreter tflite;

    private static int digit = -1;

    protected ByteBuffer imgData = null;
    private float[][] ProbArray = null;
    protected String ModelFile = "mnist.tflite";// point to model saved in assests folder

    //allocate buffer and create interface
    Classifier(Activity activity) throws IOException {
        tflite = new Interpreter(loadModelFile(activity));
        imgData = ByteBuffer.allocateDirect(INPUT_BATCH_SIZE * INPUT_HEIGHT * INPUT_WIDTH * INPUT_PIXEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        ProbArray = new float[1][10];
    }

    //load model
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, ModelFile);// load model from assets
        return tfliteModel;
    }

    //classify mat
    public void classifyMat(Mat mat) {

        long startTime = SystemClock.uptimeMillis();
        if(tflite!=null) {

            convertMattoTfLiteInput(mat);
            runInference();
        }

        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "time to run inference " + Long.toString(endTime - startTime));
    }

    //convert opencv mat to tensorflowlite input
    private void convertMattoTfLiteInput(Mat mat)
    {
        imgData.rewind();
        int pixel = 0;
        for (int i = 0; i < INPUT_HEIGHT; ++i) {
            for (int j = 0; j < INPUT_WIDTH; ++j) {
                imgData.putFloat((float)mat.get(i,j)[0]);
            }
        }
    }

    //run interface
    private void runInference() {
        if(imgData != null)
            tflite.run(imgData, ProbArray);
        digit=maxIndex(ProbArray[0]);
        Log.i(TAG, "inference " + digit);
    }

    private int maxIndex(float vector[]) {
        int maxIndex = 0;
        for (int i = 1; i < vector.length; i++) {

            if (vector[i] >= vector[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public int getDigit() {
        return digit;
    }

    //close interface
    public void close() {
        if(tflite!=null)
        {
            tflite.close();
            tflite = null;
        }
    }
}
