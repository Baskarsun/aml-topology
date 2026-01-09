from src.simulator import TransactionSimulator
from src.visualizer import graph_generator_from_dataframe, live_plot


def main():
    sim = TransactionSimulator(num_accounts=60)
    sim.generate_organic_traffic(300)
    sim.inject_fan_in('ACC_0001', num_spokes=8)
    sim.inject_fan_out('ACC_0002', num_beneficiaries=6)
    sim.inject_cycle(length=4)

    df = sim.get_dataframe()

    # generator yielding frames of the graph
    gen = graph_generator_from_dataframe(df, step=40, cumulative=True)

    # simple suspicious nodes getter using degree thresholds
    def suspicious_getter(G):
        return [n for n, d in G.in_degree() if d >= 5]

    # Run live plot for up to 12 frames with 0.6s between frames
    live_plot(gen, suspicious_nodes_getter=suspicious_getter, update_interval=0.6, max_frames=12)


if __name__ == '__main__':
    main()
