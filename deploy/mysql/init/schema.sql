CREATE DATABASE IF NOT EXISTS `scms` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `scms`;

CREATE TABLE IF NOT EXISTS `llm_user_table` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `nick_name` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`), UNIQUE KEY `uk_llm_user_nick_name` (`nick_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `llm_session_table` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `session_name` varchar(255) NOT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`), KEY `idx_llm_session_user_time` (`user_id`,`create_time`),
  CONSTRAINT `fk_llm_session_user` FOREIGN KEY (`user_id`) REFERENCES `llm_user_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `llm_chat_table` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `session_id` bigint unsigned NOT NULL,
  `user_id` bigint unsigned NOT NULL,
  `task_id` varchar(64) DEFAULT NULL,
  `question` longtext NOT NULL,
  `answer` longtext,
  `status` varchar(20) NOT NULL DEFAULT 'COMPLETED',
  `error_message` varchar(1000) DEFAULT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`), UNIQUE KEY `uk_llm_chat_task_id` (`task_id`),
  KEY `idx_llm_chat_user_session_time` (`user_id`,`session_id`,`create_time`,`id`),
  KEY `idx_llm_chat_session` (`session_id`),
  CONSTRAINT `fk_llm_chat_session` FOREIGN KEY (`session_id`) REFERENCES `llm_session_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_llm_chat_user` FOREIGN KEY (`user_id`) REFERENCES `llm_user_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `plot_info` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `session_id` bigint unsigned NOT NULL,
  `chat_id` bigint unsigned NOT NULL,
  `artifact_type` varchar(50) NOT NULL,
  `bucket_name` varchar(255) NOT NULL,
  `object_key` varchar(1024) NOT NULL,
  `etag` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`), KEY `idx_plot_chat_lookup` (`user_id`,`session_id`,`chat_id`),
  KEY `idx_plot_session` (`session_id`), KEY `idx_plot_chat` (`chat_id`),
  CONSTRAINT `fk_plot_chat` FOREIGN KEY (`chat_id`) REFERENCES `llm_chat_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_plot_session` FOREIGN KEY (`session_id`) REFERENCES `llm_session_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_plot_user` FOREIGN KEY (`user_id`) REFERENCES `llm_user_table` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `llm_chat_abstract_table` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` bigint NOT NULL,
  `session_id` bigint NOT NULL,
  `chat_start_end` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `abstract` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `users_infos` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` bigint NOT NULL,
  `user_true_name` varchar(255) DEFAULT NULL,
  `major` text, `profession` text, `interests` text, `prefer_plot_style` text,
  PRIMARY KEY (`id`), UNIQUE KEY `uk_users_infos_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
