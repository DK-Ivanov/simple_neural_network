package org.neural_network.simple_neural_network.service;

import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class LossService {

    private final NeuralNetwork neuralNetwork;
    private final DiagramService diagramService;
    private List<Double> losses = new ArrayList<>();

    public void calcEpochLoss(LearningData learningData, boolean newEpoch) {
        if (newEpoch) {
            diagramService.printLossCurve(losses.stream().reduce(Double::sum).orElseThrow());
            losses = new ArrayList<>();
            losses.add(neuralNetwork.singleLearningSession(learningData.getExample(), learningData.getAnswer()));
        } else {
            losses.add(neuralNetwork.singleLearningSession(learningData.getExample(), learningData.getAnswer()));
        }
    }



}
