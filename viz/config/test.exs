import Config

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :viz, VizWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "/ATrjiGZO3O15VVql6tW4l07S7QzUysZFJdbo3i5yjv6qPqeaB6dJSu0CkUxxwMa",
  server: false

# Print only warnings and errors during test
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

# Enable helpful, but potentially expensive runtime checks
config :phoenix_live_view,
  enable_expensive_runtime_checks: true
