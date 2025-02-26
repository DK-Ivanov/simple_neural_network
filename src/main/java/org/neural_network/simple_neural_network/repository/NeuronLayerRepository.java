package org.neural_network.simple_neural_network.repository;

import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.springframework.stereotype.Repository;

import java.util.stream.Collectors;

@Repository
@RequiredArgsConstructor
public class NeuronLayerRepository {

    private final NeuronRepository neuronRepository;

    public NeuronLayer injectWeightsAndBias(NeuronLayer layer) {
        layer.setNeurons(layer.getNeurons().parallelStream()
                .map(neuronRepository::getWeightsAndBias)
                .collect(Collectors.toList()));
        return layer;
    }

    public void add(NeuronLayer layer) {
        layer.getNeurons().parallelStream().forEach(neuronRepository::add);
    }

    public void update(NeuronLayer layer) {
        layer.getNeurons().parallelStream().forEach(neuronRepository::update);
    }

    public void clean() {
        neuronRepository.clean();
    }
}
