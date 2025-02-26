package org.neural_network.simple_neural_network.service;

import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.CheckLearningDataRepository;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.TrainingLearningDataRepository;
import org.neural_network.simple_neural_network.repository.NeuralNetworkRepository;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;


/**
 * Работа с моделью:
 *  Обучение;
 *  Получение предсказаний.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class NeuralNetworkService {

    private final NeuralNetworkRepository neuralNetworkRepository;
    private final CheckLearningDataRepository checkLearningDataRepository;
    private final TrainingLearningDataRepository trainingLearningDataRepository;
    private final NeuralNetwork neuralNetwork;
    private final LossService lossService;

    @Value("${neural-network.service.repository.rewrite-data}")
    private boolean rewrite;
    private boolean firstIteration = true;
    @Value("${neural-network.service.repository.epoch-count}")
    private int epochesCount;

//TODO:
// Создать алгоритм обучения:
// 1. происходит обучение по двум эпохам
// 2. модель предсказывает все возможные обучающие примеры после каждой эпохи:
//      * если ответы на какие-то примеры неправильные и не различаются вообще никак в обоих эпохах,
//      * то обучение сбрасывается, модель увеличивает вдвое количество этих примеров,
//      * и обучается по новой
    public void startLearning() {
        if (rewrite) {
            neuralNetworkRepository.clean();
        } else {
            neuralNetwork.setLayers(neuralNetworkRepository.injectWeightsAndBias(neuralNetwork));
        }
        long before = System.currentTimeMillis();
        int epochNumber = 0;
        int idIterator = -1;
        long startSessionTime = System.nanoTime();
        while (epochNumber < epochesCount) {
            idIterator++;
            if (trainingLearningDataRepository.hasNext(idIterator)) {
                lossService.calcEpochLoss(trainingLearningDataRepository.getById(idIterator), false);
            } else {
                idIterator = 0;
                lossService.calcEpochLoss(trainingLearningDataRepository.getById(idIterator), true);
                epochNumber++;
                if (rewrite) {
                    if (firstIteration) {
                        neuralNetworkRepository.add(neuralNetwork);
                        firstIteration = false;
                    } else {
                        neuralNetworkRepository.update(neuralNetwork);
                    }
                } else {
                    neuralNetworkRepository.update(neuralNetwork);
                }
            }
            log.info("Время сессии: {} секунд", (System.nanoTime() - startSessionTime)/ 1000_000_000d);
            startSessionTime = System.nanoTime();
        }
        long after = System.currentTimeMillis();
        log.info("Время обучения модели: {} секунд", (Double.valueOf(after - before)/1000));
        for (int i = 0; i < 10_000; i++) {
            LearningData learningData = checkLearningDataRepository.getById(i);
            log.info(
                    "Example {}: \nExpected answer: {}\n Neural network answer: {}",
                    i,
                    learningData.getAnswer(),
                    neuralNetwork.doPrediction(learningData.getExample())
            );
        }
//        if (rewrite) {
//            neuralNetworkRepository.add(neuralNetwork);
//        } else {
//            neuralNetworkRepository.update(neuralNetwork);
//        }
    }

    private void checkWeightsAndBias() {
        neuralNetwork.getLayers().stream()
                .forEach(layer -> layer.getNeurons().stream()
                        .forEach(neuron -> log.info("\nНейрон №: {} \n веса: {} \n смещение: {}",
                                neuron.getNumberInLayer(),
                                neuron.getWeight(),
                                neuron.getBias()))
                );
    }

}
