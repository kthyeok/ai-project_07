create extension if not exists vector;

create table if not exists public.documents_policy_2026 (
  id bigserial primary key,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(384) not null,
  created_at timestamptz not null default now()
);

create index if not exists documents_policy_2026_embedding_hnsw
on public.documents_policy_2026
using hnsw (embedding vector_cosine_ops);

create index if not exists documents_policy_2026_metadata_gin
on public.documents_policy_2026
using gin (metadata);

create or replace function public.match_documents_policy_2026 (
  query_embedding vector(384),
  match_count int default 8,
  filter jsonb default '{}'::jsonb
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents_policy_2026 as d
  where filter = '{}'::jsonb or d.metadata @> filter
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;

notify pgrst, 'reload schema';
