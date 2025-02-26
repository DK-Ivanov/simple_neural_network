package org.neural_network.simple_neural_network.entity.neural_network;

import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.entity.Neuron;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.neural_network.simple_neural_network.tools.loss.LossType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
@Slf4j
public abstract class NeuralNetwork {

    private final Double LEARNING_RATE;
    @Getter
    @Setter
    @NonNull
    protected List<NeuronLayer> layers;
    protected final LossType lossType;

    /**
     * Возвращаю предсказания модели (список выходных значений из последнего слоя нейронов)
     */
    public abstract List<Double> doPrediction(List<Double> featuresVector);

    /**
     * Алгоритм единичной тренировки
     */
    public Double singleLearningSession(List<Double> featuresVector, List<Double> predicatedValues) {
        List<Double> labels = doPrediction(featuresVector);
        calculateOutputLayerLossDerivative(labels, predicatedValues);
        this.layers.stream().sorted(Comparator.reverseOrder()).skip(1).forEach(NeuronLayer::calcLossDerivativeByOutput);
        this.layers = layers.stream()
                .sorted(Comparator.reverseOrder())
                .map(this::updateWeightsAndBiasOnOneLayer)
                .collect(Collectors.toList());
        this.layers = layers.stream().sorted().toList();
        return predicatedValues.stream()
                .map(predicatedValue -> lossType.getLossFunction().calcLoss(labels.get(predicatedValues.indexOf(predicatedValue)), predicatedValue))
                .reduce(Double::sum)
                .orElseThrow() / predicatedValues.size();
    }

    protected abstract void calculateOutputLayerLossDerivative(List<Double> labels,
                                                    List<Double> expectedLabels);

    /**
     * Обновление весов и смещения для всех нейронов одного слоя
     */
    private NeuronLayer updateWeightsAndBiasOnOneLayer(NeuronLayer layer) {
        layer.setNeurons(layer.getNeurons().parallelStream()
                        .map(neuron -> updateWeightsAndBias(
                                neuron,
                                layer.getLastInputData(),
                                layer.getLastLossDerivativeByOutputData().get(neuron.getNumberInLayer())
                                )
                        )
                        .collect(Collectors.toList()));
        return layer;
    }

    /**
     * Обновление весов и смещения для одного нейрона
     */
    private Neuron updateWeightsAndBias(Neuron neuron,
                                        List<Double> lastInputData,
                                        Double lossDerivativeByOutputData) {
        neuron.setWeight(neuron.getWeight().stream()
                        .map(weight -> weight
                                .setValueAndGetWeight(
                                        weight.getValue() -
                                                LEARNING_RATE *
                                                        lastInputData.get(weight.getNumberInNeuron()) *
                                                                lossDerivativeByOutputData)
                        )
                        .collect(Collectors.toList()));
        neuron.setBias(neuron.getBias() - LEARNING_RATE * lossDerivativeByOutputData);
        return neuron;
    }

}
