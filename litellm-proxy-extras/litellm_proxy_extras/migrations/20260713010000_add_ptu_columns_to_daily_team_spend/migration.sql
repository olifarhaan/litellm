-- AlterTable
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN IF NOT EXISTS "ptu_flat_cost" DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE "LiteLLM_DailyTeamSpend" ADD COLUMN IF NOT EXISTS "ptu_source_model_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_DailyTeamSpend_ptu_source_model_id_idx" ON "LiteLLM_DailyTeamSpend"("ptu_source_model_id");
