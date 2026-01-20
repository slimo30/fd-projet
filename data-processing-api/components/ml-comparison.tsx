"use client";

import { useState, useEffect } from "react";
import { BarChart, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Image from "next/image";

export default function MlComparison() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchComparison = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/ml/comparison");
      if (response.ok) {
        const result = await response.json();
        setData(result);
      }
    } catch (error) {
      console.error("Error fetching comparison:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComparison();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold tracking-tight">Model Comparison</h2>
        <Button onClick={fetchComparison} disabled={loading} variant="outline">
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {!data || !data.comparison ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center h-60 text-muted-foreground">
            <BarChart className="h-10 w-10 mb-4 opacity-20" />
            <p>No model results available yet. Run some analysis first.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Comparison Chart</CardTitle>
            </CardHeader>
            <CardContent>
              {data.plot_url ? (
                <div className="relative w-full h-[400px]">
                  <Image
                    src={data.plot_url}
                    alt="Comparison Plot"
                    fill
                    className="object-contain"
                    unoptimized
                  />
                </div>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  Not enough data for plot
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Detailed Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Algorithm</TableHead>
                    <TableHead>Primary Metric</TableHead>
                    <TableHead>Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(data.comparison).map(([algo, metrics]: [string, any]) => {
                    const isRegression = 'r2_score' in metrics;
                    const primaryMetric = isRegression ? 'R2 Score' : 'Accuracy';
                    const score = isRegression ? metrics.r2_score : metrics.accuracy;
                    
                    return (
                      <TableRow key={algo}>
                        <TableCell className="font-medium capitalize">
                          {algo.replace(/_/g, " ")}
                        </TableCell>
                        <TableCell>{primaryMetric}</TableCell>
                        <TableCell>{score?.toFixed(4) || "N/A"}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
