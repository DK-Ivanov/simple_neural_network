package org.neural_network.simple_neural_network.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.CheckLearningDataRepository;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.TrainingLearningDataRepository;
import org.neural_network.simple_neural_network.repository.NeuralNetworkRepository;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


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
//    private final LearningDataRepositoryTest trainingLearningDataRepository;
    private final TrainingLearningDataRepository trainingLearningDataRepository;
    private final NeuralNetwork neuralNetwork;
    private final LossService lossService;

    @Value("${neural-network.service.pack-size}")
    private int packSize;
    @Value("${neural-network.service.repository.rewrite-data}")
    private boolean rewrite;
    @Value("${neural-network.service.repository.epoch-count}")
    private int epochesCount;

    private int idIterator = 0;

//TODO:
// Создать алгоритм обучения:
// 1. происходит обучение по двум эпохам
// 2. модель предсказывает все возможные обучающие примеры после каждой эпохи:
//      * если ответы на какие-то примеры неправильные и не различаются вообще никак в обоих эпохах,
//      * то обучение сбрасывается, модель увеличивает вдвое количество этих примеров,
//      * и обучается по новой
    public void startLearning() {
        configureRepositoryAndNetwork();
        learning();
        checkModel();
    }

    private void configureRepositoryAndNetwork() {
        if (rewrite) {
            neuralNetworkRepository.clean();
        } else {
            neuralNetwork.setLayers(neuralNetworkRepository.injectWeightsAndBias(neuralNetwork));
        }
    }

    private void learning() {
        int epochNumber = 0;
        long before = System.currentTimeMillis();
        while (epochNumber < epochesCount) {
            learnEpoch();
            epochNumber++;
        }
        long after = System.currentTimeMillis();
        log.info("Время обучения модели: {} секунд", (Double.valueOf(after - before)/1_000));
    }

    private void checkModel() {
        for (int i = 60_000; i < 70_000; i++) {
            LearningData learningData = checkLearningDataRepository.getById(i);
            log.info(
                    "Example {}: \nExpected answer: {}\n Neural network answer: {}",
                    i,
                    learningData.getAnswer(),
                    neuralNetwork.doPrediction(learningData.getExample())
            );
        }
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

    private void learnEpoch() {
        idIterator = 0;
        while (hasNext()) {
            List<LearningData> pack = generatePack();
            Collections.shuffle(pack);
            learnPack(pack);
        }
        if (rewrite) {
            neuralNetworkRepository.add(neuralNetwork);
            rewrite = false;
        } else {
            neuralNetworkRepository.update(neuralNetwork);
        }
        lossService.printLossCurve();
    }

    private List<LearningData> generatePack() {
        List<LearningData> pack = new ArrayList<>();
        for (int i = 0; i < packSize; i++) {
            if (hasNext()) {
                pack.add(trainingLearningDataRepository.getById(idIterator));
                idIterator++;
            } else {
                return pack;
            }
        }
        return pack;
    }

    private void learnPack(List<LearningData> pack) {
        long startTime = System.nanoTime();
        pack.forEach(lossService::calcExample);
        log.info("Время сессии из {} примеров: {} секунд", pack.size(), (System.nanoTime() - startTime) / 1_000_000_000d);
    }

    private boolean hasNext() {
        return trainingLearningDataRepository.hasNext(idIterator);
    }


}
