package com.example.llmback.service.rocketService;

import com.example.llmback.entity.message.RocketMessage;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.math.BigInteger;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

@Component
public class RedisStorage {

    private static final Duration TASK_TTL = Duration.ofDays(1);
    private static final Duration PROCESSING_LOCK_TTL = Duration.ofMinutes(10);
    private final StringRedisTemplate redisTemplate;

    public RedisStorage(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public String generateMapKey(String taskId) {
        return "redis:hash:" + taskId;
    }

    private String generateProcessingLockKey(String taskId) {
        return "redis:lock:ai-task:" + taskId;
    }

    public boolean tryStartTask(String taskId) {
        Boolean acquired = redisTemplate.opsForValue().setIfAbsent(
                generateProcessingLockKey(taskId),
                "PROCESSING",
                PROCESSING_LOCK_TTL
        );
        return Boolean.TRUE.equals(acquired);
    }

    public void releaseTaskLock(String taskId) {
        redisTemplate.delete(generateProcessingLockKey(taskId));
    }

    public void restorageMessage(RocketMessage message) {
        if (message == null || message.getTaskId() == null) {
            throw new IllegalArgumentException("message and taskId must not be null");
        }
        String key = generateMapKey(message.getTaskId());
        redisTemplate.opsForHash().putAll(key, transformToMap(message));
        redisTemplate.expire(key, TASK_TTL);
    }

    public void updateStatus(String taskId, String status) {
        redisTemplate.opsForHash().put(generateMapKey(taskId), "status", status);
    }

    public void updateFailureReason(String taskId, String reason) {
        redisTemplate.opsForHash().put(
                generateMapKey(taskId), "failureReason", reason == null ? "unknown" : reason);
    }

    public long incrementRetryCount(String taskId) {
        Long count = redisTemplate.opsForHash().increment(
                generateMapKey(taskId), "retryCount", 1L
        );
        redisTemplate.expire(generateMapKey(taskId), TASK_TTL);
        return count == null ? 1L : count;
    }

    public RocketMessage loadMessage(String taskId) {
        String key = generateMapKey(taskId);
        RocketMessage message = new RocketMessage();
        message.setTaskId(taskId);
        message.setTopic(value(key, "topic"));
        message.setQuestion(value(key, "question"));
        message.setStatus(value(key, "status"));
        message.setUserId(toBigInteger(value(key, "userId")));
        message.setSessionId(toBigInteger(value(key, "sessionId")));
        message.setChatId(toBigInteger(value(key, "chatId")));
        message.setFailureReason(value(key, "failureReason"));
        return message;
    }

    private String value(String key, String field) {
        Object value = redisTemplate.opsForHash().get(key, field);
        return value == null ? null : value.toString();
    }

    private BigInteger toBigInteger(String value) {
        return value == null || value.isBlank() ? null : new BigInteger(value);
    }

    private Map<String, String> transformToMap(RocketMessage message) {
        Map<String, String> map = new HashMap<>();
        put(map, "taskId", message.getTaskId());
        put(map, "topic", message.getTopic());
        put(map, "status", message.getStatus());
        put(map, "question", message.getQuestion());
        put(map, "userId", message.getUserId());
        put(map, "sessionId", message.getSessionId());
        put(map, "chatId", message.getChatId());
        put(map, "failureReason", message.getFailureReason());
        return map;
    }

    private void put(Map<String, String> map, String key, Object value) {
        if (value != null) {
            map.put(key, value.toString());
        }
    }
}
