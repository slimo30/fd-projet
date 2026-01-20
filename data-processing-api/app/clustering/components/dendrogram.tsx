"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface DendrogramProps {
  filePath: string
}

export default function Dendrogram({ filePath }: DendrogramProps) {
  const [method, setMethod] = useState("ward")
  const [maxClusters, setMaxClusters] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setImageUrl(null)

    try {
      const response = await fetch("http://localhost:5001/clustering/dendrogram", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path: filePath,
          algorithm: "agnes",
          method,
          max_clusters: maxClusters,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to generate dendrogram")
      }

      const data = await response.json()
      // Use the full URL directly
      setImageUrl(`http://localhost:5001${data.image_url}`)
    } catch (err: any) {
      setError(err.message || "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="method">Linkage Method</Label>
          <Select value={method} onValueChange={setMethod}>
            <SelectTrigger id="method">
              <SelectValue placeholder="Select method" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ward">Ward</SelectItem>
              <SelectItem value="complete">Complete</SelectItem>
              <SelectItem value="average">Average</SelectItem>
              <SelectItem value="single">Single</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="max-clusters">Maximum Clusters</Label>
          <Input
            id="max-clusters"
            type="number"
            min={2}
            max={20}
            value={maxClusters}
            onChange={(e) => setMaxClusters(Number.parseInt(e.target.value))}
          />
        </div>
      </div>

      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          "Generate Dendrogram"
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
          <h3 className="text-lg font-medium mb-4">Hierarchical Clustering Dendrogram</h3>
          <div className="flex justify-center">
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Dendrogram"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
            />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            The dendrogram shows the hierarchical relationship between clusters. The height of each branch represents
            the distance between clusters.
          </p>
          <div className="mt-2 flex justify-center">
            <Button variant="outline" onClick={() => window.open(imageUrl, "_blank")}>
              Open in New Tab
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
