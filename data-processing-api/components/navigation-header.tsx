"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Brain, Network, Database } from "lucide-react"
import { cn } from "@/lib/utils"

const navItems = [
  {
    name: "Data Processing",
    href: "/",
    icon: Database,
  },
  {
    name: "Machine Learning",
    href: "/ml",
    icon: Brain,
  },
  {
    name: "Clustering",
    href: "/clustering",
    icon: Network,
  },
]

export function NavigationHeader() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-14 items-center">
          <div className="flex items-center gap-8 flex-1">
            <Link href="/" className="flex items-center gap-2 font-semibold text-lg">
              <div className="flex h-7 w-7 items-center justify-center rounded bg-primary">
                <Database className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="hidden sm:inline-block">Data Platform</span>
            </Link>
            
            <nav className="flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href || 
                               (item.href !== "/" && pathname.startsWith(item.href))
                
                return (
                  <Link 
                    key={item.href} 
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors",
                      "hover:bg-accent hover:text-accent-foreground",
                      isActive 
                        ? "bg-accent text-accent-foreground" 
                        : "text-muted-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden md:inline">{item.name}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </div>
    </header>
  )
}
