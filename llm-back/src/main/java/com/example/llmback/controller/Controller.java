package com.example.llmback.controller;

import com.example.llmback.dto.request.ChatRequest;
import com.example.llmback.dto.response.ChatResponse;
import com.example.llmback.entity.SessionChat;
import com.example.llmback.entity.UserSession;
import com.example.llmback.entity.response.Response;
import com.example.llmback.service.mapperService.MapperService;
import com.example.llmback.service.rocketService.MessageProducer;
import com.example.llmback.service.sqlProtection.RedisCache;
import com.example.llmback.service.sqlProtection.RedisLimiting;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.*;

import java.math.BigInteger;
import java.time.Duration;
import java.util.List;
import java.util.Objects;

@RestController
@RequestMapping("/ai")
public class Controller {

    private final RedisCache redisCache;
    private final MapperService mapperService;
    private final MessageProducer messageProducer;
    private final RedisLimiting redisLimiting;

    public Controller(RedisCache redisCache,
                      MapperService mapperService,
                      MessageProducer messageProducer,
                      RedisLimiting redisLimiting) {
        this.redisCache = redisCache;
        this.mapperService = mapperService;
        this.messageProducer = messageProducer;
        this.redisLimiting = redisLimiting;
    }

    @PostMapping("/login")
    public Response<String> login(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "password", required = false) String password,
            HttpServletRequest request
    ) throws Exception {
        if (authorization != null && authorization.startsWith("Bearer ")) {
            String token = extractToken(authorization);
            Response<BigInteger> response = redisCache.validateToken(token);
            if (Objects.equals(response.getStatus(), "200") && response.getData() != null) {
                return new Response<>("200", "登录成功", token);
            }
        }

        return redisCache.firstLogin(name, password, request.getRemoteAddr());
    }

    @PostMapping("/register")
    public Response<String> register(
            @RequestParam("name") String name,
            @RequestParam("password") String password,
            HttpServletRequest request
    ) throws Exception {
        return redisCache.register(name, password, request.getRemoteAddr());
    }

    @PostMapping("/user/update")
    public Response<String> updateName(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam("name") String name,
            @RequestParam("password") String password
    ) throws Exception {
        String token = extractToken(authorization);
        BigInteger userId = redisCache.getUserIdByToken(token);
        mapperService.updateUserName(name, userId, password);
        return new Response<>("200", "修改成功", null);
    }

    @PostMapping("/user/delete")
    public Response<String> deleteUser(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam("name") String name,
            @RequestParam("password") String password
    ) throws Exception {
        String token = extractToken(authorization);
        BigInteger userId = redisCache.getUserIdByToken(token);
        mapperService.deleteUser(userId, name, password);
        return new Response<>("200", "删除成功", null);
    }

    @GetMapping("/user/sessions")
    public Response<List<UserSession>> getSessions(
            @RequestHeader(value = "Authorization", required = false) String authorization
    ) throws Exception {
        String token = extractToken(authorization);
        return new Response<>("200", "读取成功", redisCache.getAllSession(token));
    }

    @PostMapping("/user/chat")
    public Response<ChatResponse> sendQuestion(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestBody ChatRequest request
    ) throws Exception {
        if (request == null || request.getQuestion() == null || request.getQuestion().isBlank()) {
            throw new IllegalArgumentException("question 不能为空");
        }

        String token = extractToken(authorization);
        BigInteger userId = redisCache.getUserIdByToken(token);
        String generatedKey = redisLimiting.generateChatSubmitKey(userId);

        boolean allowed = redisLimiting.limitingRequest(
                generatedKey,
                Duration.ofSeconds(20),
                2
        );

        if (!allowed){
            throw new RuntimeException("请求过于频繁");
        }

        BigInteger sessionId = request.getSessionId();
        boolean firstChat = sessionId == null;

        if (firstChat) {
            UserSession session = mapperService.firstCreate(userId, request.getQuestion());
            sessionId = session.getSessionId();
            redisCache.addSessionToCache(session);
        } else if (mapperService.getSession(userId, sessionId) == null) {
            throw new IllegalArgumentException("会话不存在或不属于当前用户");
        }

        String taskId = messageProducer.generateTaskId(userId, sessionId);
        SessionChat chat = mapperService.createGeneratingChat(
                taskId,
                sessionId,
                userId,
                request.getQuestion()
        );

        try {
            messageProducer.sendMessage(
                    taskId,
                    request.getQuestion(),
                     userId,
                     sessionId,
                     chat.getChatId()
             );
        } catch (Exception exception) {
            mapperService.markChatFailed(taskId, "任务发送失败");
            throw exception;
        }

        return new Response<>(
                "200",
                "任务提交成功",
                new ChatResponse(sessionId, chat.getChatId(), taskId, firstChat)
        );
    }

    @GetMapping("/user/chat")
    public Response<List<SessionChat>> getAllChat(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam("sessionId") BigInteger sessionId
    ) throws Exception {
        String token = extractToken(authorization);
        return new Response<>("200", "读取成功", redisCache.getAllChat(token, sessionId));
    }

    @GetMapping("/user/chat/plot")
    public Response<List<String>> getAllPlotUrls(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam("sessionId") BigInteger sessionId,
            @RequestParam("chatId") BigInteger chatId
    ) throws Exception {
        String token = extractToken(authorization);
        BigInteger userId = redisCache.getUserIdByToken(token);
        List<String> urls = redisCache.getAllPlotURLS(userId, sessionId, chatId);
        return new Response<>("200", "读取成功", urls);
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
