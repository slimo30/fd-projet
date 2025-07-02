import { UploadForm } from "@/components/upload-form"
import { DataPreview } from "@/components/data-preview"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink } from "@/components/ui/breadcrumb"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function UploadPage() {
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
              <BreadcrumbLink>Upload Data</BreadcrumbLink>
            </BreadcrumbItem>
          </Breadcrumb>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Upload Dataset</CardTitle>
                <CardDescription>Upload a CSV file containing your data for clustering analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <UploadForm />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Data Preview</CardTitle>
                <CardDescription>Preview of the uploaded dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataPreview />
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}
