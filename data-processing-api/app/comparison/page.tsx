"use client"

import { useState, useEffect } from "react"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink } from "@/components/ui/breadcrumb"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, RefreshCw } from "lucide-react"
import Image from "next/image"
import { ComparisonTable } from "@/components/comparison-table"

export default function ComparisonPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [plotUrl, setPlotUrl] = useState<string | null>(null)

  const fetchComparison = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:5000/clustering/comparison")

      if (!response.ok) {
        throw new Error("Failed to fetch comparison data")
      }

      const data = await response.json()
      setComparisonData(data.comparison)
      setPlotUrl(data.plot_url)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchComparison()
  }, [])

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold">Clustering Analysis Dashboard</h1>
          </div>
        </div>
      </header>
      <main className="flex-1">
        <div className="container py-6">
          <Breadcrumb className="mb-6">
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbLink>Comparison</BreadcrumbLink>
            </BreadcrumbItem>
          </Breadcrumb>

          <Card className="mb-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Algorithm Comparison</CardTitle>
                  <CardDescription>Compare performance metrics across different clustering algorithms</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={fetchComparison} disabled={loading}>
                  <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {error && (
                <Alert variant="destructive" className="mb-6">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {!error && !comparisonData && !loading && (
                <Alert className="mb-6">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>No comparison data available</AlertTitle>
                  <AlertDescription>
                    Run at least two different clustering algorithms to see a comparison.
                  </AlertDescription>
                </Alert>
              )}

              {comparisonData && (
                <div className="space-y-6">
                  <ComparisonTable data={comparisonData} />

                  {plotUrl && (
                    <div className="mt-6">
                      <h3 className="text-lg font-medium mb-2">Comparison Visualization</h3>
                      <div className="border rounded-md p-2 bg-white">
                        <Image
                          src={`http://localhost:5000${plotUrl}`}
                          alt="Algorithm Comparison Plot"
                          width={800}
                          height={400}
                          className="mx-auto"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
