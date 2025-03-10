package org.neural_network.simple_neural_network.mnist.entity;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MnistDataReader {

    public MnistMatrix[] readData(String dataFilePath, String labelFilePath) throws IOException {

        try (DataInputStream dataInputStream = new DataInputStream(
                new BufferedInputStream(new FileInputStream(dataFilePath)));
             DataInputStream labelInputStream = new DataInputStream(
                     new BufferedInputStream(new FileInputStream(labelFilePath)))) {

            int magicNumber = dataInputStream.readInt();
            int numberOfItems = dataInputStream.readInt();
            int nRows = dataInputStream.readInt();
            int nCols = dataInputStream.readInt();

            System.out.println("magic number is " + magicNumber);
            System.out.println("number of items is " + numberOfItems);
            System.out.println("number of rows is: " + nRows);
            System.out.println("number of cols is: " + nCols);

            int labelMagicNumber = labelInputStream.readInt();
            int numberOfLabels = labelInputStream.readInt();

            System.out.println("labels magic number is: " + labelMagicNumber);
            System.out.println("number of labels is: " + numberOfLabels);

            MnistMatrix[] data = new MnistMatrix[numberOfItems];

            assert numberOfItems == numberOfLabels;

            for (int i = 0; i < numberOfItems; i++) {
                MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);
                mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
                for (int r = 0; r < nRows; r++) {
                    for (int c = 0; c < nCols; c++) {
                        mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                    }
                }
                data[i] = mnistMatrix;
            }
            return data;
        }
    }
}