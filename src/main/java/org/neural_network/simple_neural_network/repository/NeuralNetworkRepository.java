package org.neural_network.simple_neural_network.repository;

import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.entity.neural_network.NeuralNetwork;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Collectors;

@Repository
@RequiredArgsConstructor
public class NeuralNetworkRepository {

    private final NeuronLayerRepository neuronLayerRepository;

    public List<NeuronLayer> injectWeightsAndBias(NeuralNetwork network) {
        return network.getLayers().stream()
                .map(neuronLayerRepository::injectWeightsAndBias)
                .collect(Collectors.toList());

    }

    public void add(NeuralNetwork network) {
        network.getLayers().forEach(neuronLayerRepository::add);
    }

    public void update(NeuralNetwork network) {
        network.getLayers().forEach(neuronLayerRepository::update);
    }

    public void clean() {
        neuronLayerRepository.clean();
    }
}
