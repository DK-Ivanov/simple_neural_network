package org.neural_network.simple_neural_network.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class LossService {

    private final NeuralNetwork neuralNetwork;
    private final DiagramService diagramService;
    private List<BigDecimal> losses = new ArrayList<>();

    public void calcExample(LearningData learningData) {
            losses.add(neuralNetwork.singleLearningSession(learningData.getExample(), learningData.getAnswer()));
//            log.info("Loss: {}", losses.getLast());
    }

    public void printLossCurve() {
        diagramService.printLossCurve(losses.stream().reduce(BigDecimal::add).orElseThrow());
        losses = new ArrayList<>();
    }



}
