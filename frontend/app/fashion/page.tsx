import { SearchApp } from "@/components/SearchApp";

export default function FashionPage() {
  return (
    <div className="flex flex-col gap-8">
      <div className="mx-auto max-w-[1680px]">
        <h1 className="font-display text-4xl font-black tracking-tight">Busqueda de moda</h1>
        <p className="mt-1 text-sm text-muted">
          Busca por descripcion (texto) o subiendo una imagen para comparar por similitud SIFT.
        </p>
      </div>
      <SearchApp
        app="fashion"
        mediaKind="image"
        textLabel="Por descripcion"
        textPlaceholder="Ej: red dress, denim jacket..."
        fileLabel="Por imagen"
        fileAccept="image/*"
      />
    </div>
  );
}
