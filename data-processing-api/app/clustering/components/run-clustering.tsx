"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent } from "@/components/ui/card";

interface RunClusteringProps {
  filePath: string;
  algorithm: string;
}

interface PerformanceMetrics {
  silhouette?: number | null;
  davies_bouldin?: number | null;
  calinski_harabasz?: number | null;
}

export default function RunClustering({
  filePath,
  algorithm,
}: RunClusteringProps) {
  const [k, setK] = useState(3);
  const [method, setMethod] = useState("ward");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setImageUrl(null);
    setMetrics(null);

    try {
      const payload: any = {
        path: filePath,
        algorithm,
      };

      if (algorithm === "agnes" || algorithm === "diana") {
        payload.n_clusters = k;
        payload.method = method;
      } else {
        payload.k = k;
      }

      const response = await fetch("http://localhost:5001/clustering/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Clustering failed");

      const data = await response.json();
      setMetrics(data.performance);
      setImageUrl(`http://localhost:5001${data.plot_url}`);
      onComplete(data.performance);
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  // Format metric value with null/undefined check
  const formatMetric = (value: number | undefined | null): string => {
    return value !== undefined && value !== null ? value.toFixed(4) : "N/A";
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {algorithm === "agnes" || algorithm === "diana" ? (
          <>
            <div className="space-y-2">
              <Label htmlFor="n-clusters">Number of Clusters</Label>
              <Input
                id="n-clusters"
                type="number"
                min={2}
                max={20}
                value={k}
                onChange={(e) => setK(Number.parseInt(e.target.value))}
              />
            </div>
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
          </>
        ) : (
          <div className="space-y-2 col-span-2">
            <Label htmlFor="k">Number of Clusters (K)</Label>
            <Input
              id="k"
              type="number"
              min={2}
              max={20}
              value={k}
              onChange={(e) => setK(Number.parseInt(e.target.value))}
            />
          </div>
        )}
      </div>

      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Running...
          </>
        ) : (
          `Run ${algorithm.toUpperCase()} Clustering`
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
          <h3 className="text-lg font-medium mb-4">Clustering Results</h3>
          <div className="flex justify-center">
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Clustering Results"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
            />
          </div>
          <div className="mt-2 flex justify-center">
            <Button
              variant="outline"
              onClick={() => window.open(imageUrl, "_blank")}
            >
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
                <p className="text-sm text-muted-foreground">
                  Silhouette Score
                </p>
                <p className="text-2xl font-bold mt-2">
                  {formatMetric(metrics.silhouette)}
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Range: [-1, 1] | Higher is better
                </p>
              </div>
              <div className="border rounded-md p-4 text-center">
                <p className="text-sm text-muted-foreground">
                  Davies-Bouldin Index
                </p>
                <p className="text-2xl font-bold mt-2">
                  {formatMetric(metrics.davies_bouldin)}
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Range: [0, ∞) | Lower is better
                </p>
              </div>
              <div className="border rounded-md p-4 text-center">
                <p className="text-sm text-muted-foreground">
                  Calinski-Harabasz Index
                </p>
                <p className="text-2xl font-bold mt-2">
                  {formatMetric(metrics.calinski_harabasz)}
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Range: [0, ∞) | Higher is better
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
function onComplete(performance: any) {
  throw new Error("Function not implemented.");
}

