import { SearchApp } from "@/components/SearchApp";

export default function MusicPage() {
  return (
    <div className="flex flex-col gap-8">
      <div className="mx-auto max-w-[1680px]">
        <h1 className="font-display text-4xl font-black tracking-tight">Busqueda musical</h1>
        <p className="mt-1 text-sm text-muted">
          Busca por letra (texto) o subiendo un clip de audio para comparar por similitud MFCC.
        </p>
      </div>
      <SearchApp
        app="music"
        mediaKind="audio"
        textLabel="Por letra"
        textPlaceholder="Ej: love, heartbreak, summer..."
        fileLabel="Por audio"
        fileAccept="audio/*"
      />
    </div>
  );
}
