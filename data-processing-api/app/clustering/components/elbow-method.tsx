"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface ElbowMethodProps {
  filePath: string
  algorithm: string
}

export default function ElbowMethod({ filePath, algorithm }: ElbowMethodProps) {
  const [maxK, setMaxK] = useState(10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setImageUrl(null)

    try {
      const response = await fetch("http://localhost:5001/clustering/elbow", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path: filePath,
          algorithm,
          max_k: maxK,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to generate elbow plot")
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
      <div className="space-y-2">
        <Label htmlFor="max-k">Maximum number of clusters (K)</Label>
        <Input
          id="max-k"
          type="number"
          min={2}
          max={20}
          value={maxK}
          onChange={(e) => setMaxK(Number.parseInt(e.target.value))}
        />
      </div>

      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          "Generate Elbow Plot"
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
          <h3 className="text-lg font-medium mb-4">Elbow Method Plot</h3>
          <div className="flex justify-center">
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Elbow Method Plot"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
            />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            The elbow point indicates the optimal number of clusters. Look for the point where the curve starts to
            flatten.
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
