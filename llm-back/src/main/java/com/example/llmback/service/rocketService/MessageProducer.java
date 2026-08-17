package com.example.llmback.service.rocketService;

import com.example.llmback.entity.message.RocketMessage;
import lombok.extern.slf4j.Slf4j;
import org.apache.rocketmq.client.apis.ClientServiceProvider;
import org.apache.rocketmq.client.apis.message.Message;
import org.apache.rocketmq.client.apis.producer.Producer;
import org.apache.rocketmq.client.apis.producer.SendReceipt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

import java.math.BigInteger;
import java.util.UUID;

@Slf4j
@Component
public class MessageProducer {

    @Value("${rocketmq.task-topic}")
    private String topic;

    @Value("${rocketmq.task-tag}")
    private String tag;

    @Value("${rocketmq.producer.max-attempts:3}")
    private int maxAttempts;

    private final Producer producer;
    private final ClientServiceProvider provider;
    private final ObjectMapper objectMapper;
    private final RedisStorage redisStorage;

    public MessageProducer(Producer producer, ClientServiceProvider provider,
                           ObjectMapper objectMapper, RedisStorage redisStorage) {
        this.producer = producer;
        this.provider = provider;
        this.objectMapper = objectMapper;
        this.redisStorage = redisStorage;
    }

    public void sendMessage(String taskId, String question, BigInteger userId,
                            BigInteger sessionId, BigInteger chatId) throws Exception {
        RocketMessage rocketMessage = new RocketMessage();
        rocketMessage.setTopic(topic);
        rocketMessage.setTaskId(taskId);
        rocketMessage.setQuestion(question);
        rocketMessage.setUserId(userId);
        rocketMessage.setSessionId(sessionId);
        rocketMessage.setChatId(chatId);
        rocketMessage.setStatus("CREATED");

        try {
            redisStorage.restorageMessage(rocketMessage);
            Message message = provider.newMessageBuilder()
                    .setTopic(topic)
                    .setKeys(taskId)
                    .setBody(objectMapper.writeValueAsBytes(rocketMessage))
                    .setTag(tag)
                    .build();

            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                SendReceipt receipt = producer.send(message);
                if (receipt != null && receipt.getMessageId() != null) {
                    log.info("RocketMQ message sent, taskId={}, attempt={}", taskId, attempt);
                    return;
                }
            }
            throw new IllegalStateException("RocketMQ message send failed");
        } catch (Exception e) {
            redisStorage.updateStatus(taskId, "FAILURE");
            redisStorage.updateFailureReason(taskId,
                    e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage());
            log.error("RocketMQ message send failed, taskId={}", taskId, e);
            throw e;
        }
    }

    public String generateTaskId(BigInteger userId, BigInteger sessionId) {
        return "user:" + userId + ":session:" + sessionId + ":" + UUID.randomUUID();
    }
}
