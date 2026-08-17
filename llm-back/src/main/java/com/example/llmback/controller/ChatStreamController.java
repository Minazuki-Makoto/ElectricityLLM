package com.example.llmback.controller;

import com.example.llmback.entity.SessionChat;
import com.example.llmback.entity.stream.ChatStreamEvent;
import com.example.llmback.service.mapperService.MapperService;
import com.example.llmback.service.sqlProtection.RedisCache;
import com.example.llmback.service.streamSave.RedisStreamLoad;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.task.TaskExecutor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.math.BigInteger;
import java.util.List;

@RestController
@RequestMapping("/ai/user/chat")
public class ChatStreamController {
    private final RedisStreamLoad redisStreamLoad;
    private final RedisCache redisCache;
    private final TaskExecutor taskExecutor;
    private final long sseTimeoutMillis;

    public ChatStreamController(
            RedisStreamLoad redisStreamLoad,
            RedisCache redisCache,
            @Qualifier("chatStreamExecutor") TaskExecutor taskExecutor,
            @Value("${app.sse.timeout-millis:600000}") long sseTimeoutMillis
    ) {
        this.redisStreamLoad = redisStreamLoad;
        this.redisCache = redisCache;
        this.taskExecutor = taskExecutor;
        this.sseTimeoutMillis = sseTimeoutMillis;
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamChat(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam("taskId") String taskId
    ) throws Exception {
        if (taskId == null || taskId.isBlank()) {
            throw new IllegalArgumentException("taskId 不能为空");
        }

        String token = extractToken(authorization);
        BigInteger userId = redisCache.getUserIdByToken(token);

        SseEmitter emitter = new SseEmitter(sseTimeoutMillis);
        taskExecutor.execute(() -> redisStreamLoad.sendSSE(emitter, taskId, userId));
        return emitter;
    }

    private String extractToken(String authorization) {
        if (authorization == null || !authorization.startsWith("Bearer ")) {
            throw new IllegalArgumentException("Authorization 请求头格式错误");
        }

        String token = authorization.substring(7).strip();
        if (token.isEmpty()) {
            throw new IllegalArgumentException("token 不能为空");
        }
        return token;
    }
}
