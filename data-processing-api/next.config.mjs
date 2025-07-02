const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    domains: ['127.0.0.1'],
  },
  async rewrites() {
    return [
      {
        source: '/static/:path*',
        destination: 'http://127.0.0.1:5000/static/:path*',
      },
    ];
  },
}

export default nextConfig
