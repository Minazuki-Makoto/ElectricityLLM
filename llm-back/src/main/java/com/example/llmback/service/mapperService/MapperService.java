package com.example.llmback.service.mapperService;

import com.example.llmback.entity.PlotInfo;
import com.example.llmback.entity.SessionChat;
import com.example.llmback.entity.User;
import com.example.llmback.entity.UserSession;
import com.example.llmback.mapper.ChatMapper;
import com.example.llmback.mapper.PlotMapper;
import com.example.llmback.mapper.SessionMapper;
import com.example.llmback.mapper.UserMapper;
import com.example.llmback.service.minioService.ParsePlotURL;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigInteger;
import java.time.LocalDateTime;
import java.util.*;
@Service
public class MapperService {

    private final ChatMapper chatMapper;
    private final SessionMapper sessionMapper;
    private final UserMapper userMapper;
    private final PlotMapper plotMapper;
    private final ParsePlotURL parsePlotURL;

    public MapperService(ChatMapper chatMapper,
                         SessionMapper sessionMapper,
                         UserMapper userMapper,
                         PlotMapper plotMapper,
                         ParsePlotURL parsePlotURL){
        this.chatMapper = chatMapper;
        this.sessionMapper = sessionMapper;
        this.userMapper = userMapper;
        this.plotMapper = plotMapper;
        this.parsePlotURL = parsePlotURL;
    }

    public User getUser(
            String name,
            String password
    ){
        return userMapper.getUser(
                name,
                password
        );
    }

    public List<UserSession> getAllSession(
            BigInteger userId
    ){
        return sessionMapper.getAllSession(userId);
    }

    public UserSession getSession(
            BigInteger userId,
            BigInteger sessionId
    ){
        return sessionMapper.getSession(
                userId,
                sessionId
        );
    }

    public List<SessionChat> getAllChat(
            BigInteger userId,
            BigInteger sessionId
    ){
        return chatMapper.getAllChat(
                userId,
                sessionId
        );
    }

    public int addUser(
            String name,
            String password
    )
    {
        return userMapper.addUser(
                name,
                password
        );
    }

    public boolean userNameExists(String name) {
        return userMapper.countByName(name) > 0;
    }

    public SessionChat addChat(
            BigInteger sessionId,
            BigInteger userId,
            String question,
            String answer
    ){
        LocalDateTime now = LocalDateTime.now();
        SessionChat chat = new SessionChat();
        chat.setSessionId(sessionId);
        chat.setUserId(userId);
        chat.setQuestion(question);
        chat.setAnswer(answer);
        chat.setCreateTime(now);
        chat.setUpdateTime(now);

        int affectedRows = chatMapper.chatInsert(chat);
        if (affectedRows != 1) {
            throw new IllegalStateException("新增聊天记录失败");
        }
        return chat;
    }

    public SessionChat createGeneratingChat(
            String taskId,
            BigInteger sessionId,
            BigInteger userId,
            String question
    ) {
        SessionChat chat = new SessionChat();
        chat.setSessionId(sessionId);
        chat.setUserId(userId);
        chat.setQuestion(question);

        int affectedRows = chatMapper.insertGenerating(chat, taskId);
        if (affectedRows != 1 || chat.getChatId() == null) {
            throw new IllegalStateException("创建聊天任务失败");
        }
        return chat;
    }

    public SessionChat getChatByTaskId(String taskId, BigInteger userId) {
        return chatMapper.getByTaskIdAndUserId(taskId, userId);
    }

    @Transactional
    public SessionChat completeChat(String taskId, String answer) {
        int affectedRows = chatMapper.markCompleted(taskId, answer);
        if (affectedRows != 1) {
            throw new IllegalStateException("完成聊天任务失败");
        }

        SessionChat chat = chatMapper.getByTaskId(taskId);
        if (chat == null) {
            throw new IllegalStateException("聊天任务不存在");
        }
        return chat;
    }

    public int markChatFailed(String taskId, String errorMessage) {
        return chatMapper.markFailed(taskId, errorMessage);
    }

    public UserSession firstCreate(BigInteger userId,
                           String question){
        String normalizedQuestion = question.strip();
        String[] parts = normalizedQuestion.split("[？?；;。！!\\r\\n]", 2);
        String name = parts[0].isBlank() ? normalizedQuestion : parts[0];
        if (name.length() > 30) {
            name = name.substring(0, 30);
        }
        UserSession session = new UserSession();
        session.setUserId(userId);
        session.setSessionName(name);
        LocalDateTime now = LocalDateTime.now();
        session.setCreateTime(now);
        session.setUpdateTime(now);
        sessionMapper.insertSession(session);

        return session;
    }


    public int updateUserName(
            String name,
            BigInteger userId,
            String password
    ){
        return userMapper.updateName(
                name,
                userId,
                password
        );
    }

    public int deleteUser(
            BigInteger userId,
            String name,
            String password
    ){
        return userMapper.deleteUser(
                userId,
                name,
                password
        );
    }

    public List<String> getURLS(BigInteger userId,
                                BigInteger sessionId,
                                BigInteger chatId){
        return plotMapper.selectByChatId(userId, sessionId, chatId)
                .stream()
                .map(info -> parsePlotURL.generateURL(
                        info.getBucketName(),
                        info.getObjectKey()
                ))
                .toList();
    }

    @Transactional
    public int insertPlots(BigInteger userId,
                           BigInteger sessionId,
                           BigInteger chatId,
                           List<PlotInfo> artifacts){
        int insertedRows = 0;
        for (PlotInfo artifact : artifacts) {
            artifact.setUserId(userId);
            artifact.setSessionId(sessionId);
            artifact.setChatId(chatId);
            insertedRows += plotMapper.insert(artifact);
        }
        return insertedRows;
    }
}
