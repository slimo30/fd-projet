"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Card, CardContent } from "@/components/ui/card"

interface RunDBSCANProps {
  filePath: string
}

interface PerformanceMetrics {
  silhouette?: number
  davies_bouldin?: number
  calinski_harabasz?: number
}

export default function RunDBSCAN({ filePath }: RunDBSCANProps) {
  const [eps, setEps] = useState(0.5)
  const [minSamples, setMinSamples] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setImageUrl(null)
    setMetrics(null)

    try {
      const response = await fetch("http://localhost:5001/clustering/dbscan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path: filePath,
          eps,
          min_samples: minSamples,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to run DBSCAN clustering")
      }

      const data = await response.json()
      // Use the full URL directly
      setImageUrl(`http://localhost:5001${data.plot_url}`)
      setMetrics(data.performance)
    } catch (err: any) {
      setError(err.message || "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="eps">Epsilon (ε): {eps}</Label>
            <span className="text-sm text-muted-foreground">Maximum distance between points</span>
          </div>
          <div className="flex items-center gap-4">
            <Slider id="eps" min={0.1} max={2} step={0.1} value={[eps]} onValueChange={(value) => setEps(value[0])} />
            <Input
              type="number"
              min={0.1}
              max={2}
              step={0.1}
              value={eps}
              onChange={(e) => setEps(Number.parseFloat(e.target.value))}
              className="w-20"
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="min-samples">Min Samples: {minSamples}</Label>
            <span className="text-sm text-muted-foreground">Minimum points to form a cluster</span>
          </div>
          <div className="flex items-center gap-4">
            <Slider
              id="min-samples"
              min={2}
              max={20}
              step={1}
              value={[minSamples]}
              onValueChange={(value) => setMinSamples(value[0])}
            />
            <Input
              type="number"
              min={2}
              max={20}
              step={1}
              value={minSamples}
              onChange={(e) => setMinSamples(Number.parseInt(e.target.value))}
              className="w-20"
            />
          </div>
        </div>
      </div>

      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Running...
          </>
        ) : (
          "Run DBSCAN Clustering"
        )}
      </Button>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {imageUrl && (
        <div className="mt-6 border rounded-lg p-4">
          <h3 className="text-lg font-medium mb-4">DBSCAN Clustering Results</h3>
          <div className="flex justify-center">
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="DBSCAN Clustering Results"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
            />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            DBSCAN identifies clusters of arbitrary shape and can detect noise points (shown in a different color).
          </p>
          <div className="mt-2 flex justify-center">
            <Button variant="outline" onClick={() => window.open(imageUrl, "_blank")}>
              Open in New Tab
            </Button>
          </div>
        </div>
      )}

      {metrics && (
        <Card>
          <CardContent className="pt-6">
            <h3 className="text-lg font-medium mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="border rounded-md p-4 text-center">
                <p className="text-sm text-muted-foreground">Silhouette Score</p>
                <p className="text-2xl font-bold mt-2">
                  {metrics.silhouette !== undefined ? metrics.silhouette.toFixed(4) : "N/A"}
                </p>
                <p className="text-xs text-muted-foreground mt-2">Range: [-1, 1] | Higher is better</p>
              </div>
              <div className="border rounded-md p-4 text-center">
                <p className="text-sm text-muted-foreground">Davies-Bouldin Index</p>
                <p className="text-2xl font-bold mt-2">
                  {metrics.davies_bouldin !== undefined ? metrics.davies_bouldin.toFixed(4) : "N/A"}
                </p>
                <p className="text-xs text-muted-foreground mt-2">Range: [0, ∞) | Lower is better</p>
              </div>
              <div className="border rounded-md p-4 text-center">
                <p className="text-sm text-muted-foreground">Calinski-Harabasz Index</p>
                <p className="text-2xl font-bold mt-2">
                  {metrics.calinski_harabasz !== undefined ? metrics.calinski_harabasz.toFixed(4) : "N/A"}
                </p>
                <p className="text-xs text-muted-foreground mt-2">Range: [0, ∞) | Higher is better</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
