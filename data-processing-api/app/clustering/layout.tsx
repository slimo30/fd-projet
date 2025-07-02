import type React from "react"
export default function ClusteringLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto py-4">
          <h1 className="text-xl font-bold">Clustering Analysis Tool</h1>
        </div>
      </header>
      <main>{children}</main>
    </div>
  )
}
