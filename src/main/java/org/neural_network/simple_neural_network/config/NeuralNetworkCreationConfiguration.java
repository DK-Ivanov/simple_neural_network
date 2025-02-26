package org.neural_network.simple_neural_network.config;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.neural_network.simple_neural_network.entity.Neuron;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.neural_network.simple_neural_network.entity.neural_network.SimpleNeuralNetwork;
import org.neural_network.simple_neural_network.entity.neural_network.SoftmaxNeuralNetwork;
import org.neural_network.simple_neural_network.tools.activate_function.SimpleActivateFunc;
import org.neural_network.simple_neural_network.tools.loss.LossType;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Slf4j
@Configuration
@Setter
public class NeuralNetworkCreationConfiguration {
    @Value("${neural-network.learning-rate}")
    private Double learningRate;
    @Value("${neural-network.layers.neurons-count}")
    private List<Integer> layersNeuronsCount;
    @Value("${neural-network.layers.features-count}")
    private int featuresCount;
    @Value("${neural-network.layers.activate-func}")
    private List<Integer> activateFunc;
    @Value("${neural-network.loss.loss-type}")
    private int lossType;

    @Bean
    public NeuralNetwork neuralNetwork() {
        List<NeuronLayer> neuronLayers = new ArrayList<>();
        AtomicInteger neuronId = new AtomicInteger(1);
        for (int i = 0; i < layersNeuronsCount.size(); i++) {
            int featuresCount = i == 0 ? this.featuresCount : layersNeuronsCount.get(i-1);

            AtomicInteger neuronCounter = new AtomicInteger(0);
//            if (activateFunc.get(i) == 4)
            SimpleActivateFunc layersActivateFunc = SimpleActivateFunc.getByNumber(activateFunc.get(i));
            NeuronLayer thisNeuronLayer = new NeuronLayer(
                    i,
                    Stream.generate(() -> new Neuron(
                                    neuronCounter.getAndAdd(1),
                                    featuresCount,
                                    neuronId.getAndAdd(1),
                                    layersActivateFunc
                            ))
                            .limit(layersNeuronsCount.get(i))
                            .collect(Collectors.toList()) ,
                    featuresCount
            );
            if (i != 0) {
                NeuronLayer previousLayer = neuronLayers.getLast();
                thisNeuronLayer.setPrevious(previousLayer);
                previousLayer.setNext(thisNeuronLayer);
            }
            neuronLayers.add(thisNeuronLayer);
        }
        if (activateFunc.stream().limit(activateFunc.size() - 1).anyMatch(integer -> integer.equals(4))) {
            throw new RuntimeException("Функция активации softmax может находиться только на последнем слое!");
        }
        if (activateFunc.getLast().equals(4)) {
            return new SoftmaxNeuralNetwork(
                    learningRate,
                    neuronLayers,
                    LossType.getByNumber(lossType)
            );
        }
        return new SimpleNeuralNetwork(
                learningRate,
                neuronLayers,
                LossType.getByNumber(lossType)
        );
    }
}
