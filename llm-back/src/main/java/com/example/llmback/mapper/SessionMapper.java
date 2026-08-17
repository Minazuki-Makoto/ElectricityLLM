package com.example.llmback.mapper;

import com.example.llmback.entity.UserSession;
import org.apache.ibatis.annotations.*;

import java.math.BigInteger;
import java.util.List;

@Mapper
public interface SessionMapper {

    @Select("""
SELECT id AS sessionId,
       user_id AS userId,
       session_name AS sessionName,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_session_table
WHERE user_id = #{userId}
""")
    List<UserSession> getAllSession(
            @Param("userId")BigInteger userId
            );

    @Select("""
SELECT id AS sessionId,
       user_id AS userId,
       session_name AS sessionName,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_session_table
WHERE user_id = #{userId}
AND id = #{sessionId}
""")
    UserSession getSession(
            @Param("userId") BigInteger userId,
            @Param("sessionId") BigInteger sessionId
    );

    @Insert("""
INSERT INTO llm_session_table (user_id,session_name,create_time,update_time)
VALUES (
        #{userId},
        #{sessionName},
        Now(),
        Now()
)
""")
    @Options(useGeneratedKeys = true, keyProperty = "sessionId", keyColumn = "id")
    int insertSession(UserSession session);

    @Update("""
UPDATE llm_session_table
SET session_name = #{sessionName},
    update_time = Now()
WHERE user_id = #{userId}
AND id = #{sessionId}
""")
    int update(
            @Param("sessionName") String sessionName,
            @Param("userId") BigInteger userId,
            @Param("sessionId") BigInteger sessionId
    );

}
