import Link from "next/link";

const TICKER: { name: string; color: string; ms: number }[] = [
  { name: "SPIMI", color: "var(--engine-spimi)", ms: 81 },
  { name: "GIN", color: "var(--engine-gin)", ms: 55 },
  { name: "GiST", color: "var(--engine-gist)", ms: 62 },
  { name: "PGVECTOR", color: "var(--engine-pgvector)", ms: 4369 },
];

export default function Home() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-14">
      <div className="scanlines flex flex-col gap-5 rounded-lg py-10">
        <span className="font-mono text-xs uppercase tracking-[0.16em] text-muted">
          Cuatro motores · una misma consulta
        </span>
        <h1 className="font-display max-w-4xl text-6xl leading-[0.98] font-black tracking-tight sm:text-7xl">
          Compara motores de busqueda en tiempo real
        </h1>
        <p className="max-w-2xl text-base text-muted">
          GRID enfrenta el motor propio (SPIMI + TF-IDF/histogramas de similitud coseno) contra las
          capacidades nativas de PostgreSQL (pgvector HNSW, GIN, GiST) sobre dos dominios:
          busqueda musical (letras y audio) y busqueda de moda (descripcion e imagen). Quien gana
          lo decide el tiempo.
        </p>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border pt-5">
          {TICKER.map((t) => (
            <span key={t.name} className="flex items-center gap-2 font-mono text-sm">
              <span className="h-2 w-2 rounded-full" style={{ background: t.color }} />
              <span className="font-semibold" style={{ color: t.color }}>
                {t.name}
              </span>
              <span className="text-muted tabular-nums">{t.ms}ms</span>
            </span>
          ))}
          <span className="font-mono text-[11px] text-muted">ejemplo · corre tu propia query abajo</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <AppCard
          index="01"
          href="/music"
          title="Musica"
          description="Busca canciones por letra (texto) o subiendo un clip de audio (MFCC)."
        />
        <AppCard
          index="02"
          href="/fashion"
          title="Fashion"
          description="Busca productos por descripcion (texto) o subiendo una imagen (SIFT)."
        />
      </div>
    </div>
  );
}

function AppCard({
  index,
  href,
  title,
  description,
}: {
  index: string;
  href: string;
  title: string;
  description: string;
}) {
  return (
    <Link
      href={href}
      className="group relative flex flex-col justify-between gap-8 overflow-hidden rounded-xl border border-border bg-surface p-8 transition-colors hover:border-engine-spimi"
    >
      <span className="font-display pointer-events-none absolute -top-6 -right-2 text-[9rem] leading-none font-black text-foreground/[0.04] select-none">
        {index}
      </span>
      <div className="relative flex items-start justify-between">
        <span className="font-mono text-xs text-muted">{index}</span>
        <span className="font-mono text-muted transition-transform group-hover:translate-x-1 group-hover:text-engine-spimi">
          →
        </span>
      </div>
      <div className="relative flex flex-col gap-3">
        <span className="font-display text-3xl font-bold tracking-tight group-hover:text-engine-spimi">
          {title}
        </span>
        <span className="max-w-sm text-sm text-muted">{description}</span>
        <div className="mt-2 flex gap-1.5">
          {TICKER.map((t) => (
            <span key={t.name} className="h-1.5 w-1.5 rounded-full" style={{ background: t.color }} />
          ))}
        </div>
      </div>
    </Link>
  );
}
