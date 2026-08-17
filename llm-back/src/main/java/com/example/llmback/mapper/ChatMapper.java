package com.example.llmback.mapper;

import com.example.llmback.entity.SessionChat;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Options;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.math.BigInteger;
import java.util.*;
@Mapper
public interface ChatMapper {

    @Select("""

            SELECT id AS chatId,
       session_id AS sessionId,
       user_id AS userId,
       question,
       answer,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_chat_table
WHERE user_id = #{userId}
AND session_id = #{sessionId}
ORDER BY create_time ASC, id ASC
""")
    List<SessionChat> getAllChat(
            @Param("userId") BigInteger userId,
            @Param("sessionId") BigInteger sessionId
    );

    @Insert("""

INSERT INTO llm_chat_table (session_id,user_id,question,answer,create_time,update_time)
VALUES (
        #{sessionId},
        #{userId},
        #{question},
        #{answer},
        Now(),
        Now()
)
""")
    @Options(useGeneratedKeys = true, keyProperty = "chatId", keyColumn = "id")
    int chatInsert(
            SessionChat chat
    );

    @Insert("""

INSERT INTO llm_chat_table (
       session_id,user_id,task_id,question,answer,status,error_message,create_time,update_time
)
VALUES (
       #{chat.sessionId}, #{chat.userId}, #{taskId}, #{chat.question},
       NULL, 'GENERATING', NULL, Now(), Now()
)
""")
    @Options(useGeneratedKeys = true, keyProperty = "chat.chatId", keyColumn = "id")
    int insertGenerating(
            @Param("chat") SessionChat chat,
            @Param("taskId") String taskId
    );

    @Select("""

SELECT id AS chatId,
       session_id AS sessionId,
       user_id AS userId,
       question,
       answer,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_chat_table
WHERE task_id = #{taskId}
  AND user_id = #{userId}
""")
    SessionChat getByTaskIdAndUserId(
            @Param("taskId") String taskId,
            @Param("userId") BigInteger userId
    );

    @Select("""

SELECT id AS chatId,
       session_id AS sessionId,
       user_id AS userId,
       question,
       answer,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_chat_table
WHERE task_id = #{taskId}
""")
    SessionChat getByTaskId(@Param("taskId") String taskId);

    @Update("""

UPDATE llm_chat_table
SET answer = #{answer},
    status = 'COMPLETED',
    error_message = NULL,
    update_time = Now()
WHERE task_id = #{taskId}
""")
    int markCompleted(
            @Param("taskId") String taskId,
            @Param("answer") String answer
    );

    @Update(
    """
    UPDATE llm_chat_table
    SET status = 'FAILED',
    error_message = #{errorMessage},
    update_time = Now()
    WHERE task_id = #{taskId}
    AND status <> 'COMPLETED'
            """
 )
    int markFailed(
            @Param("taskId") String taskId,
            @Param("errorMessage") String errorMess
    );
}