package com.example.llmback.config;

import com.example.llmback.service.rocketService.MessageConsumer;
import org.apache.rocketmq.client.apis.ClientConfiguration;
import org.apache.rocketmq.client.apis.ClientException;
import org.apache.rocketmq.client.apis.ClientServiceProvider;
import org.apache.rocketmq.client.apis.consumer.FilterExpression;
import org.apache.rocketmq.client.apis.consumer.FilterExpressionType;
import org.apache.rocketmq.client.apis.consumer.PushConsumer;
import org.apache.rocketmq.client.apis.producer.Producer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Collections;

@Configuration
public class RocketConfig {

    @Value("${rocketmq.endpoints}")
    private String endpoints;

    @Value("${rocketmq.task-topic}")
    private String taskTopic;

    @Value("${rocketmq.consumer-group}")
    private String consumerGroup;

    @Value("${rocketmq.task-tag}")
    private String taskTag;

    @Bean
    public ClientServiceProvider clientServiceProvider() {
        return ClientServiceProvider.loadService();
    }

    @Bean
    public ClientConfiguration clientConfiguration() {
        return ClientConfiguration.newBuilder().setEndpoints(endpoints).build();
    }

    @Bean(destroyMethod = "close")
    public Producer buildProducer(ClientServiceProvider provider,
                                  ClientConfiguration configuration) throws ClientException {
        return provider.newProducerBuilder()
                .setClientConfiguration(configuration)
                .setTopics(taskTopic)
                .build();
    }

    @Bean(destroyMethod = "close")
    public PushConsumer pushConsumer(ClientServiceProvider provider,
                                     ClientConfiguration configuration,
                                     MessageConsumer messageConsumer) throws ClientException {
        FilterExpression filter = new FilterExpression(taskTag, FilterExpressionType.TAG);
        return provider.newPushConsumerBuilder()
                .setClientConfiguration(configuration)
                .setConsumerGroup(consumerGroup)
                .setSubscriptionExpressions(Collections.singletonMap(taskTopic, filter))
                .setMessageListener(messageConsumer)
                .build();
    }
}
