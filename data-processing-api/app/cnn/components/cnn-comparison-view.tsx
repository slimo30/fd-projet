"use client"

import { useState, useEffect } from "react"
import { Loader2, RefreshCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cnnApi } from "@/lib/api"

export default function CNNComparison() {
    const [loading, setLoading] = useState(false)
    const [data, setData] = useState<any>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetchComparison()
    }, [])

    const fetchComparison = async () => {
        try {
            setLoading(true)
            setError(null)
            const result = await cnnApi.compareDatasets()
            setData(result)
        } catch (err) {
            setError("Train at least one model to see comparisons.")
        } finally {
            setLoading(false)
        }
    }

    if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin" /></div>

    if (error) {
        return (
            <div className="text-center py-12 space-y-4">
                <p className="text-muted-foreground">{error}</p>
                <Button onClick={fetchComparison} variant="outline">
                    <RefreshCcw className="mr-2 h-4 w-4" /> Try Again
                </Button>
            </div>
        )
    }

    if (!data) return null

    return (
        <div className="space-y-6">
            <div className="flex justify-end">
                <Button onClick={fetchComparison} variant="outline" size="sm">
                    <RefreshCcw className="mr-2 h-4 w-4" /> Refresh
                </Button>
            </div>

            <div className="border rounded-lg p-4 bg-white">
                <h4 className="font-semibold mb-4 text-center">Performance Comparison</h4>
                <div className="relative w-full aspect-[21/9] min-h-[300px]">
                    <img
                        src={`http://127.0.0.1:5001${data.plot_url}`}
                        alt="Comparison"
                        className="rounded-md object-contain w-full h-full"
                    />
                </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(data.comparison).map(([key, metrics]: [string, any]) => (
                    <div key={key} className="border rounded-lg p-4 space-y-3">
                        <h5 className="font-semibold capitalize border-b pb-2">
                            {key.replace('cnn_', '').replace('_', ' ')}
                        </h5>
                        <dl className="space-y-1 text-sm">
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Accuracy</span>
                                <span className="font-bold text-green-600">{(metrics.accuracy * 100).toFixed(2)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Loss</span>
                                <span>{metrics.val_loss.toFixed(4)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Params</span>
                                <span className="font-mono">{metrics.total_params.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Epochs</span>
                                <span>{metrics.epochs_trained}</span>
                            </div>
                        </dl>
                    </div>
                ))}
            </div>
        </div>
    )
}
