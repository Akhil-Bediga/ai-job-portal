-- AI Job Portal: structured job store (MySQL)
-- The unstructured side (full descriptions, extracted skills, resumes) lives in
-- MongoDB. Here we keep the queryable, structured columns the API filters on.

CREATE DATABASE IF NOT EXISTS jobportal
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE jobportal;

DROP TABLE IF EXISTS job_skills;
DROP TABLE IF EXISTS jobs;

CREATE TABLE jobs (
  job_id           VARCHAR(36)  NOT NULL,
  title            VARCHAR(160) NOT NULL,
  company          VARCHAR(120) NOT NULL,
  family           VARCHAR(60)  NOT NULL,          -- job family (analytics + eval label)
  city             VARCHAR(80),
  state            VARCHAR(40),
  remote           TINYINT(1)   NOT NULL DEFAULT 0,
  employment_type  ENUM('FULL_TIME','PART_TIME','CONTRACT') NOT NULL DEFAULT 'FULL_TIME',
  seniority        ENUM('JUNIOR','MID','SENIOR','LEAD') NOT NULL DEFAULT 'MID',
  salary_min       INT UNSIGNED,
  salary_max       INT UNSIGNED,
  salary_avg       INT UNSIGNED,
  posted_date      DATE,
  PRIMARY KEY (job_id),
  KEY ix_jobs_family (family),
  KEY ix_jobs_seniority (seniority),
  KEY ix_jobs_state (state),
  KEY ix_jobs_title (title)
) ENGINE=InnoDB;

-- one row per (job, skill); lets the API answer "which jobs need Spark?" fast
CREATE TABLE job_skills (
  id        BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  job_id    VARCHAR(36)     NOT NULL,
  skill     VARCHAR(80)     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_job_skill (job_id, skill),
  KEY ix_skill (skill),
  CONSTRAINT fk_skill_job FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB;
