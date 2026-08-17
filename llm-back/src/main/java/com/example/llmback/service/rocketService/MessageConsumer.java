package com.example.llmback.service.rocketService;

import com.example.llmback.entity.message.RocketMessage;
import com.example.llmback.service.FullService;
import lombok.extern.slf4j.Slf4j;
import org.apache.rocketmq.client.apis.consumer.ConsumeResult;
import org.apache.rocketmq.client.apis.consumer.MessageListener;
import org.apache.rocketmq.client.apis.message.MessageView;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;
import tools.jackson.databind.ObjectMapper;

import java.nio.ByteBuffer;

@Slf4j
@Component
public class MessageConsumer implements MessageListener {

    private final ObjectMapper objectMapper;
    private final FullService fullService;
    private final RedisStorage redisStorage;

    @Value("${rocketmq.consumer.max-retry-attempts:3}")
    private int maxRetryAttempts;

    public MessageConsumer(ObjectMapper objectMapper,
                           FullService fullService,
                           RedisStorage redisStorage) {
        this.objectMapper = objectMapper;
        this.fullService = fullService;
        this.redisStorage = redisStorage;
    }

    @Override
    public ConsumeResult consume(MessageView messageView) {
        RocketMessage message = null;
        try {
            ByteBuffer body = messageView.getBody();
            byte[] bytes = new byte[body.remaining()];
            body.get(bytes);
            message = objectMapper.readValue(bytes, RocketMessage.class);
            validate(message);

            if (!redisStorage.tryStartTask(message.getTaskId())) {
                log.warn("Ignoring duplicate RocketMQ delivery, taskId={}, sessionId={}",
                        message.getTaskId(), message.getSessionId());
                return ConsumeResult.SUCCESS;
            }

            redisStorage.updateStatus(message.getTaskId(), "PROCESSING");
            fullService.processMessage(message);
            redisStorage.updateStatus(message.getTaskId(), "SUCCESS");

            log.info("RocketMQ task processed, taskId={}, sessionId={}",
                    message.getTaskId(), message.getSessionId());
            return ConsumeResult.SUCCESS;
        } catch (Exception e) {
            String taskId = message == null ? null : message.getTaskId();
            log.error("RocketMQ task failed, taskId={}", taskId, e);
            if (taskId == null || taskId.isBlank()) {
                log.error("Discarding invalid RocketMQ message without taskId");
                return ConsumeResult.SUCCESS;
            }

            redisStorage.releaseTaskLock(taskId);
            long retryCount = redisStorage.incrementRetryCount(taskId);
            redisStorage.updateFailureReason(taskId, safeMessage(e));

            if (retryCount >= maxRetryAttempts) {
                redisStorage.updateStatus(taskId, "FAILURE");
                fullService.markFailed(taskId, safeMessage(e));
                log.error(
                        "RocketMQ task reached retry limit and will be acknowledged, taskId={}, attempts={}",
                        taskId, retryCount
                );
                return ConsumeResult.SUCCESS;
            }

            redisStorage.updateStatus(taskId, "RETRYING");
            log.warn(
                    "RocketMQ task will retry, taskId={}, attempt={}, maxAttempts={}",
                    taskId, retryCount, maxRetryAttempts
            );
            return ConsumeResult.FAILURE;
        }
    }

    private void validate(RocketMessage message) {
        if (message == null || message.getTaskId() == null || message.getTaskId().isBlank()) {
            throw new IllegalArgumentException("taskId must not be blank");
        }
        if (message.getQuestion() == null || message.getQuestion().isBlank()) {
            throw new IllegalArgumentException("question must not be blank");
        }
        if (message.getUserId() == null || message.getSessionId() == null) {
            throw new IllegalArgumentException("userId and sessionId must not be null");
        }
    }

    private String safeMessage(Exception e) {
        return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
    }
}
