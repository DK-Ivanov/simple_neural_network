package org.neural_network.simple_neural_network.mnist.repository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.mnist.entity.MnistDataReader;
import org.neural_network.simple_neural_network.mnist.entity.MnistMatrix;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.CheckLearningDataRepository;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.LearningDataRepository;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.TrainingLearningDataRepository;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.stereotype.Repository;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Repository
@RequiredArgsConstructor
@Slf4j
public class MNISTDataRepository {
    private static final String FILE_DATA_60K = "src/main/resources/mnist/training/train-images-idx3-ubyte";
    private static final String FILE_LABELS_60K = "src/main/resources/mnist/training/train-labels-idx1-ubyte";
    private static final String FILE_DATA_10K = "src/main/resources/mnist/tests/t10k-images-idx3-ubyte";
    private static final String FILE_LABELS_10K = "src/main/resources/mnist/tests/t10k-labels-idx1-ubyte";

    private final TrainingLearningDataRepository trainingLearningDataRepository;
    private final CheckLearningDataRepository checkLearningDataRepository;

    public void rewriteTrainingData() {
        rewriteData(FILE_DATA_60K, FILE_LABELS_60K, trainingLearningDataRepository);
    }

    public void rewriteCheckData() {
        rewriteData(FILE_DATA_10K, FILE_LABELS_10K, checkLearningDataRepository);
    }

    private void rewriteData(String fileData, String fileLabels, LearningDataRepository learningDataRepository) {
        List<Double> zeroList784 = Stream.generate(() -> 0d).limit(784).collect(Collectors.toList());
        try {
            MnistMatrix[] mnistMatrix = new MnistDataReader().readData(fileData, fileLabels);
            for (int i = 0; i < mnistMatrix.length; i++) {
                MnistMatrix matrix = mnistMatrix[i];
                List<Double> x = new ArrayList<>(zeroList784); //784

                for (int r = 0; r < matrix.getNumberOfRows(); r++) {
                    for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                        x.set(r * matrix.getNumberOfColumns() + c, Double.valueOf(matrix.getValue(r, c) / 255));
                    }
                }

                List<Double> y = Stream.generate(() -> 0d).limit(10).collect(Collectors.toList());
                y.set(matrix.getLabel(), 1d);
                LearningData learningData = new LearningData(y, x);
                learningDataRepository.add(learningData);
                log.info("Example {} successfully uploaded!", learningData.getId());
                System.gc();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
