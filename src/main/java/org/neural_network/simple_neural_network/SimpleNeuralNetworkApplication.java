package org.neural_network.simple_neural_network;


import org.neural_network.simple_neural_network.mnist.repository.MNISTDataRepository;
import org.neural_network.simple_neural_network.repository.learning_data_reposes.LearningDataRepositoryTest;
import org.neural_network.simple_neural_network.service.NeuralNetworkService;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

import java.util.List;

@SpringBootApplication
public class SimpleNeuralNetworkApplication {


	public static void main(String[] args) {
		SpringApplicationBuilder builder = new SpringApplicationBuilder(SimpleNeuralNetworkApplication.class);
		builder.headless(false);
		ConfigurableApplicationContext context = builder.run(args);
		NeuralNetworkService neuralNetworkService = context.getBean(NeuralNetworkService.class);
		neuralNetworkService.startLearning();
	}

}
