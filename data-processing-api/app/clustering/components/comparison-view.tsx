"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Loader2, AlertCircle, RefreshCw } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface ComparisonData {
  [algorithm: string]: {
    silhouette?: number
    davies_bouldin?: number
    calinski_harabasz?: number
  }
}

export default function ComparisonView() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null)

  const fetchComparison = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:5001/clustering/comparison")

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to fetch comparison data")
      }

      const data = await response.json()
      // Use the full URL directly if plot_url exists
      setImageUrl(data.plot_url ? `http://localhost:5001${data.plot_url}` : null)
      setComparisonData(data.comparison || null)
    } catch (err: any) {
      setError(err.message || "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchComparison()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Button
          variant="outline"
          size="sm"
          onClick={fetchComparison}
          disabled={loading}
          className="flex items-center gap-2"
        >
          <RefreshCw size={16} />
          Refresh Comparison
        </Button>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading comparison data...</span>
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {!loading && !error && !comparisonData && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No comparison data available</AlertTitle>
          <AlertDescription>Run at least two different clustering algorithms to see a comparison.</AlertDescription>
        </Alert>
      )}

      {imageUrl && (
        <div className="mt-6 border rounded-lg p-4">
          <h3 className="text-lg font-medium mb-4">Algorithm Comparison Chart</h3>
          <div className="flex justify-center">
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Algorithm Comparison"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
            />
          </div>
          <div className="mt-2 flex justify-center">
            <Button variant="outline" onClick={() => window.open(imageUrl, "_blank")}>
              Open in New Tab
            </Button>
          </div>
        </div>
      )}

      {comparisonData && Object.keys(comparisonData).length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-4">Performance Metrics Comparison</h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Algorithm</TableHead>
                <TableHead>Silhouette Score</TableHead>
                <TableHead>Davies-Bouldin Index</TableHead>
                <TableHead>Calinski-Harabasz Index</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(comparisonData).map(([algorithm, metrics]) => (
                <TableRow key={algorithm}>
                  <TableCell className="font-medium">{algorithm.toUpperCase()}</TableCell>
                  <TableCell>{metrics.silhouette !== undefined ? metrics.silhouette.toFixed(4) : "N/A"}</TableCell>
                  <TableCell>
                    {metrics.davies_bouldin !== undefined ? metrics.davies_bouldin.toFixed(4) : "N/A"}
                  </TableCell>
                  <TableCell>
                    {metrics.calinski_harabasz !== undefined ? metrics.calinski_harabasz.toFixed(4) : "N/A"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div className="mt-4 text-sm text-muted-foreground">
            <p>
              <strong>Silhouette Score:</strong> Range [-1, 1] - Higher is better. Measures how similar an object is to
              its own cluster compared to other clusters.
            </p>
            <p>
              <strong>Davies-Bouldin Index:</strong> Range [0, ∞) - Lower is better. Evaluates clustering based on the
              average similarity between clusters.
            </p>
            <p>
              <strong>Calinski-Harabasz Index:</strong> Range [0, ∞) - Higher is better. Ratio of between-cluster
              variance to within-cluster variance.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
