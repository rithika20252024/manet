/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * resilient_manet_ns3.cc
 * =======================
 * NS-3 Simulation Script for RESILIENT-MANET Data Collection
 * Area: 1000m x 1000m | Nodes: 100 | Time: 40s | CBR Traffic: 512B, 4 pkts/s
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/energy-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ResilientManetSimulation");

int main(int argc, char *argv[]) {
    uint32_t numNodes = 100;
    double simTime = 40.0; // seconds
    std::string phyMode("DsssRate2Mbps");

    CommandLine cmd;
    cmd.AddValue("numNodes", "Number of mobile nodes", numNodes);
    cmd.AddValue("simTime", "Total simulation duration", simTime);
    cmd.Parse(argc, argv);

    // 1. Create Node Container
    NodeContainer nodes;
    nodes.Create(numNodes);

    // 2. Configure 802.11b Wireless Channel and PHY Layer
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiChannel.AddPropagationLoss("ns3::TwoRayGroundPropagationLossModel");
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(16.02)); // 250m Transmission Range
    wifiPhy.Set("TxPowerEnd", DoubleValue(16.02));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue(phyMode),
                                "ControlMode", StringValue(phyMode));

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // 3. Mobility Model: Random Waypoint over 1000m x 1000m Area
    MobilityHelper mobility;
    ObjectFactory pos;
    pos.SetTypeId("ns3::RandomRectanglePositionAllocator");
    pos.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"));
    pos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"));
    Ptr<PositionAllocator> taPositionAlloc = pos.Create()->GetObject<PositionAllocator>();

    mobility.SetPositionAllocator(taPositionAlloc);
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                              "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=10.0]"),
                              "Pause", StringValue("ns3::ConstantRandomVariable[Constant=5.0]"),
                              "PositionAllocator", PointerValue(taPositionAlloc));
    mobility.Install(nodes);

    // 4. Energy Model: 15.1 Joules per Node
    BasicEnergySourceHelper basicSourceHelper;
    basicSourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(15.1));
    EnergySourceContainer sources = basicSourceHelper.Install(nodes);

    WifiRadioEnergyModelHelper radioEnergyHelper;
    radioEnergyHelper.Set("TxCurrentA", DoubleValue(0.0175)); // 50 nJ/bit Tx equivalent
    radioEnergyHelper.Set("RxCurrentA", DoubleValue(0.0175)); // 50 nJ/bit Rx equivalent
    DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install(devices, sources);

    // 5. Internet & Routing Stack Installation
    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    // 6. Application Traffic: Constant Bit Rate (CBR) UDP at 4 packets/sec, 512 bytes
    uint16_t port = 9;
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(interfaces.GetAddress(1), port)));
    onoff.SetConstantRate(DataRate("16kbps"), 512); // 4 packets/sec * 512B * 8 = 16.384 kbps
    ApplicationContainer apps = onoff.Install(nodes.Get(0));
    apps.Start(Seconds(1.0));
    apps.Stop(Seconds(simTime));

    PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    apps = sink.Install(nodes.Get(1));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(simTime));

    // 7. Enable Traces & FlowMonitor
    wifiPhy.EnableAsciiAll(AsciiTraceHelper().CreateFileStream("ns3_scripts/resilient_manet_trace.tr"));
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->SerializeToXmlFile("ns3_scripts/flow_results.xml", true, true);
    Simulator::Destroy();
    return 0;
}
