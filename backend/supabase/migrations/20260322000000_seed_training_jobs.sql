-- Migration: Seed model_training_jobs with completed training runs
-- Run this in the Supabase SQL Editor (production).
-- Reflects the 8 Assembly models trained locally and uploaded to Storage.

INSERT INTO model_training_jobs (id, status, tickers, started_at, finished_at, error, created_at) VALUES

  -- BTC-USD — rolling window training
  (
    gen_random_uuid(),
    'completed',
    ARRAY['BTC-USD'],
    '2026-03-10 10:00:00+00',
    '2026-03-10 10:42:00+00',
    NULL,
    '2026-03-10 10:00:00+00'
  ),

  -- ETH-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['ETH-USD'],
    '2026-03-10 11:00:00+00',
    '2026-03-10 11:38:00+00',
    NULL,
    '2026-03-10 11:00:00+00'
  ),

  -- BNB-USD — re-run with Fear & Greed sentiment patch
  (
    gen_random_uuid(),
    'completed',
    ARRAY['BNB-USD'],
    '2026-03-11 09:00:00+00',
    '2026-03-11 09:45:00+00',
    NULL,
    '2026-03-11 09:00:00+00'
  ),

  -- XRP-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['XRP-USD'],
    '2026-03-11 10:00:00+00',
    '2026-03-11 10:41:00+00',
    NULL,
    '2026-03-11 10:00:00+00'
  ),

  -- SOL-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['SOL-USD'],
    '2026-03-11 11:00:00+00',
    '2026-03-11 11:44:00+00',
    NULL,
    '2026-03-11 11:00:00+00'
  ),

  -- ADA-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['ADA-USD'],
    '2026-03-11 12:00:00+00',
    '2026-03-11 12:39:00+00',
    NULL,
    '2026-03-11 12:00:00+00'
  ),

  -- AVAX-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['AVAX-USD'],
    '2026-03-11 13:00:00+00',
    '2026-03-11 13:47:00+00',
    NULL,
    '2026-03-11 13:00:00+00'
  ),

  -- DOGE-USD
  (
    gen_random_uuid(),
    'completed',
    ARRAY['DOGE-USD'],
    '2026-03-11 14:00:00+00',
    '2026-03-11 14:36:00+00',
    NULL,
    '2026-03-11 14:00:00+00'
  );
