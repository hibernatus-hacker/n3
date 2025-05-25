defmodule NeuroEvolution.MixProject do
  use Mix.Project

  def project do
    [
      app: :neuro_evolution,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.7"},
      {:exla, "~> 0.7"},
      {:axon, "~> 0.6"},
      {:jason, "~> 1.4"}
    ]
  end
end