package org.neural_network.simple_neural_network.repository.learning_data_reposes;


import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * Репозиторий для получения обучающих примеров для модели.
 */
@Repository
public class TrainingLearningDataRepository extends LearningDataRepository {

    public TrainingLearningDataRepository(JdbcTemplate jdbcTemplate) {
        super(jdbcTemplate, "learning_data_training");
    }


}
