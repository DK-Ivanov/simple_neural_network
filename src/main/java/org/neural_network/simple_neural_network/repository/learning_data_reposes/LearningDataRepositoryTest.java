package org.neural_network.simple_neural_network.repository.learning_data_reposes;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class LearningDataRepositoryTest extends LearningDataRepository {
    public LearningDataRepositoryTest(JdbcTemplate jdbcTemplate) {
        super(jdbcTemplate, "learning_data_test");
    }
}
