import React, { useEffect, useState, useCallback } from 'react';
import {
    View, Text, ScrollView, TouchableOpacity, StyleSheet,
    StatusBar, ActivityIndicator, RefreshControl, Alert,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { fetchOrganizedItems, fetchInsights, wearItem, washItem } from './src/api/wardrobeApi';
import COLORS from './src/theme/colors';

// ── Constants ────────────────────────────────────────────────────────────────

const ZONES = [
    { occasion: 'Home',      label: '🏠 Daily Essentials', bg: COLORS.zoneHome      },
    { occasion: 'Casual',    label: '👕 Casual Wear',       bg: COLORS.zoneCasual    },
    { occasion: 'Office',    label: '👔 Office Wear',        bg: COLORS.zoneOffice    },
    { occasion: 'Religious', label: '🙏 Occasion Wear',      bg: COLORS.zoneReligious },
    { occasion: 'Wedding',   label: '💍 Special Events',     bg: COLORS.zoneWedding   },
];

const TABS = [
    { label: 'Wardrobe', icon: '👗' },
    { label: 'Insights',  icon: '💡' },
    { label: 'Laundry',   icon: '🧺' },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

const scoreColor = (score) => {
    if (score > 70) return COLORS.scoreGood;
    if (score > 40) return COLORS.scoreMid;
    return COLORS.scoreLow;
};

// ── Sub-components ───────────────────────────────────────────────────────────

const ItemCard = ({ item, onWear }) => {
    const score = Math.round(item.sustainability_score ?? 0);
    return (
        <TouchableOpacity
            style={[styles.card, item.status === 'Dirty' && styles.dirtyCard]}
            onPress={() => onWear(item)}
            activeOpacity={0.8}
        >
            <View style={styles.cardHeader}>
                <Text style={styles.itemCategory} numberOfLines={1}>{item.category}</Text>
                <View style={[
                    styles.statusBadge,
                    { backgroundColor: item.status === 'Clean' ? COLORS.clean : COLORS.dirty },
                ]}>
                    <Text style={[
                        styles.statusText,
                        { color: item.status === 'Clean' ? COLORS.cleanText : COLORS.dirtyText },
                    ]}>
                        {item.status}
                    </Text>
                </View>
            </View>

            <Text style={styles.cardDetail}>{item.fabric} · {item.color}</Text>

            <View style={styles.wearRow}>
                <Text style={styles.wearLabel}>Cycle</Text>
                <Text style={styles.wearCount}>
                    {item.current_cycle_wears}/{item.max_wears_before_wash}
                </Text>
            </View>

            <View style={styles.scoreRow}>
                <Text style={styles.scoreLabel}>Health</Text>
                <View style={styles.scoreBarBg}>
                    <View style={[
                        styles.scoreBarFill,
                        { width: `${score}%`, backgroundColor: scoreColor(score) },
                    ]} />
                </View>
                <Text style={[styles.scoreNum, { color: scoreColor(score) }]}>{score}%</Text>
            </View>

            <Text style={styles.groupBadge}>Zone {item.layout_group}</Text>
        </TouchableOpacity>
    );
};

const StatCard = ({ value, label, bg }) => (
    <View style={[styles.statCard, { backgroundColor: bg }]}>
        <Text style={styles.statNum}>{value}</Text>
        <Text style={styles.statLabel}>{label}</Text>
    </View>
);

// ── Screens ──────────────────────────────────────────────────────────────────

const WardrobeScreen = ({ items, refreshing, onRefresh, onWear }) => (
    <ScrollView
        style={styles.screen}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.primary} />}
    >
        <Text style={styles.screenTitle}>My Wardrobe</Text>
        <Text style={styles.screenSub}>
            {items.length} items · {items.filter(i => i.status === 'Dirty').length} need washing
        </Text>

        {ZONES.map(zone => {
            const zoneItems = items.filter(i => i.occasion === zone.occasion);
            if (zoneItems.length === 0) return null;
            return (
                <View key={zone.occasion} style={[styles.zone, { backgroundColor: zone.bg }]}>
                    <View style={styles.zoneHeader}>
                        <Text style={styles.zoneTitle}>{zone.label}</Text>
                        <Text style={styles.zoneCount}>{zoneItems.length} items</Text>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                        {zoneItems.map(item => (
                            <ItemCard key={item.id} item={item} onWear={onWear} />
                        ))}
                    </ScrollView>
                </View>
            );
        })}
        <View style={{ height: 24 }} />
    </ScrollView>
);

const InsightsScreen = ({ items, insights, refreshing, onRefresh }) => {
    if (!insights) return <ActivityIndicator style={styles.centered} color={COLORS.primary} />;

    const underusedItems = items.filter(i => insights.underused?.includes(i.id));
    const overusedItems  = items.filter(i => insights.overused?.includes(i.id));
    const cleanCount     = items.filter(i => i.status === 'Clean').length;

    return (
        <ScrollView
            style={styles.screen}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.primary} />}
        >
            <Text style={styles.screenTitle}>Smart Insights</Text>
            <Text style={styles.screenSub}>Powered by ML anomaly detection</Text>

            {/* Stats */}
            <View style={styles.statsRow}>
                <StatCard value={items.length} label="Total"   bg="#E3F2FD" />
                <StatCard value={cleanCount}   label="Clean"   bg={COLORS.clean} />
                <StatCard value={insights.dirty_count} label="Dirty" bg={COLORS.dirty} />
            </View>

            {/* Laundry fatigue alert */}
            {insights.dirty_count >= 5 && (
                <View style={styles.alertCard}>
                    <Text style={styles.alertTitle}>🧺 Laundry Fatigue Alert</Text>
                    <Text style={styles.alertBody}>
                        {insights.dirty_count} items are waiting to be washed. Time for a wash cycle!
                    </Text>
                </View>
            )}

            {/* Underused */}
            {underusedItems.length > 0 && (
                <View style={styles.insightSection}>
                    <Text style={styles.insightSectionTitle}>😴 Underutilized Items</Text>
                    <Text style={styles.insightSectionSub}>
                        Worn far less than others — style them today!
                    </Text>
                    {underusedItems.map(item => (
                        <View key={item.id} style={styles.insightRow}>
                            <View style={{ flex: 1 }}>
                                <Text style={styles.insightItemName}>{item.category}</Text>
                                <Text style={styles.insightItemDetail}>
                                    {item.fabric} · {item.total_wear_count} total wears
                                </Text>
                            </View>
                            <Text style={[styles.insightScore, { color: COLORS.scoreGood }]}>
                                {Math.round(item.sustainability_score)}%
                            </Text>
                        </View>
                    ))}
                </View>
            )}

            {/* Overused */}
            {overusedItems.length > 0 && (
                <View style={styles.insightSection}>
                    <Text style={styles.insightSectionTitle}>⚠️ Overused Items</Text>
                    <Text style={styles.insightSectionSub}>
                        Worn far more than average — consider a quality check.
                    </Text>
                    {overusedItems.map(item => (
                        <View key={item.id} style={styles.insightRow}>
                            <View style={{ flex: 1 }}>
                                <Text style={styles.insightItemName}>{item.category}</Text>
                                <Text style={styles.insightItemDetail}>
                                    {item.fabric} · {item.total_wear_count} total wears
                                </Text>
                            </View>
                            <Text style={[styles.insightScore, { color: COLORS.scoreLow }]}>
                                {Math.round(item.sustainability_score)}%
                            </Text>
                        </View>
                    ))}
                </View>
            )}

            {underusedItems.length === 0 && overusedItems.length === 0 && (
                <View style={styles.emptyState}>
                    <Text style={styles.emptyEmoji}>✅</Text>
                    <Text style={styles.emptyText}>All wear patterns look normal!</Text>
                </View>
            )}
            <View style={{ height: 24 }} />
        </ScrollView>
    );
};

const LaundryScreen = ({ items, refreshing, onRefresh, onWash }) => {
    const dirtyItems = items.filter(i => i.status === 'Dirty');
    return (
        <ScrollView
            style={styles.screen}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.primary} />}
        >
            <Text style={styles.screenTitle}>Laundry Queue</Text>
            <Text style={styles.screenSub}>{dirtyItems.length} items need washing</Text>

            {dirtyItems.length === 0 ? (
                <View style={styles.emptyState}>
                    <Text style={styles.emptyEmoji}>✅</Text>
                    <Text style={styles.emptyText}>All items are clean!</Text>
                </View>
            ) : (
                dirtyItems.map(item => (
                    <View key={item.id} style={styles.laundryRow}>
                        <View style={{ flex: 1 }}>
                            <Text style={styles.laundryName}>{item.category}</Text>
                            <Text style={styles.laundryDetail}>{item.fabric} · {item.color}</Text>
                            <Text style={styles.laundryWears}>
                                Worn {item.current_cycle_wears}/{item.max_wears_before_wash} times this cycle
                            </Text>
                        </View>
                        <TouchableOpacity style={styles.washBtn} onPress={() => onWash(item)}>
                            <Text style={styles.washBtnText}>Washed ✓</Text>
                        </TouchableOpacity>
                    </View>
                ))
            )}
            <View style={{ height: 24 }} />
        </ScrollView>
    );
};

// ── Root App ─────────────────────────────────────────────────────────────────

const App = () => {
    const [activeTab,  setActiveTab]  = useState(0);
    const [items,      setItems]      = useState([]);
    const [insights,   setInsights]   = useState(null);
    const [loading,    setLoading]    = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error,      setError]      = useState(null);

    const loadData = useCallback(async () => {
        setError(null);
        try {
            const [organizedItems, smartInsights] = await Promise.all([
                fetchOrganizedItems(),
                fetchInsights(),
            ]);
            setItems(organizedItems);
            setInsights(smartInsights);
        } catch (e) {
            setError('Could not reach the server.\nMake sure the backend is running:\n\npython -m uvicorn main:app --host 0.0.0.0 --port 8000');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    useEffect(() => { loadData(); }, [loadData]);

    const onRefresh = useCallback(() => { setRefreshing(true); loadData(); }, [loadData]);

    const handleWear = useCallback(async (item) => {
        if (item.status === 'Dirty') {
            Alert.alert('Item Dirty', 'Wash this item before wearing it again.');
            return;
        }
        try {
            await wearItem(item.item_id_str);
            loadData();
        } catch {
            Alert.alert('Error', 'Could not record wear.');
        }
    }, [loadData]);

    const handleWash = useCallback(async (item) => {
        try {
            await washItem(item.item_id_str);
            loadData();
        } catch {
            Alert.alert('Error', 'Could not mark as washed.');
        }
    }, [loadData]);

    const content = () => {
        if (loading) {
            return (
                <View style={styles.loadingScreen}>
                    <ActivityIndicator size="large" color={COLORS.primary} />
                    <Text style={styles.loadingText}>Loading your wardrobe...</Text>
                </View>
            );
        }

        if (error) {
            return (
                <View style={styles.loadingScreen}>
                    <Text style={{ fontSize: 40, marginBottom: 16 }}>⚠️</Text>
                    <Text style={[styles.loadingText, { color: '#C62828', textAlign: 'center', paddingHorizontal: 32 }]}>
                        {error}
                    </Text>
                    <TouchableOpacity
                        style={{ marginTop: 24, backgroundColor: COLORS.primary, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 24 }}
                        onPress={() => { setLoading(true); loadData(); }}
                    >
                        <Text style={{ color: '#FFF', fontWeight: '600' }}>Retry</Text>
                    </TouchableOpacity>
                </View>
            );
        }

        return (
            <SafeAreaView style={styles.container}>
                <StatusBar backgroundColor={COLORS.topBar} barStyle="light-content" />

                <View style={styles.topBar}>
                    <Text style={styles.appTitle}>Smart Wardrobe</Text>
                    <Text style={styles.appSubtitle}>Sri Lanka Edition</Text>
                </View>

                <View style={{ flex: 1 }}>
                    {activeTab === 0 && (
                        <WardrobeScreen
                            items={items} refreshing={refreshing}
                            onRefresh={onRefresh} onWear={handleWear}
                        />
                    )}
                    {activeTab === 1 && (
                        <InsightsScreen
                            items={items} insights={insights}
                            refreshing={refreshing} onRefresh={onRefresh}
                        />
                    )}
                    {activeTab === 2 && (
                        <LaundryScreen
                            items={items} refreshing={refreshing}
                            onRefresh={onRefresh} onWash={handleWash}
                        />
                    )}
                </View>

                <View style={styles.tabBar}>
                    {TABS.map((tab, i) => (
                        <TouchableOpacity
                            key={i}
                            style={styles.tabItem}
                            onPress={() => setActiveTab(i)}
                            activeOpacity={0.7}
                        >
                            <Text style={styles.tabIcon}>{tab.icon}</Text>
                            <Text style={[styles.tabLabel, activeTab === i && styles.tabLabelActive]}>
                                {tab.label}
                            </Text>
                            {activeTab === i && <View style={styles.tabIndicator} />}
                        </TouchableOpacity>
                    ))}
                </View>
            </SafeAreaView>
        );
    };

    return (
        <SafeAreaProvider>
            {content()}
        </SafeAreaProvider>
    );
};

// ── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
    container:      { flex: 1, backgroundColor: COLORS.background },
    loadingScreen:  { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: COLORS.background },
    loadingText:    { marginTop: 12, color: COLORS.primary, fontSize: 16 },
    centered:       { marginTop: 60 },

    topBar:         { backgroundColor: COLORS.topBar, paddingHorizontal: 20, paddingVertical: 14 },
    appTitle:       { color: '#FFF', fontSize: 22, fontWeight: 'bold' },
    appSubtitle:    { color: COLORS.topBarSub, fontSize: 12 },

    screen:         { flex: 1, paddingHorizontal: 16, paddingTop: 16 },
    screenTitle:    { fontSize: 22, fontWeight: 'bold', color: COLORS.textPrimary, marginBottom: 2 },
    screenSub:      { fontSize: 13, color: COLORS.primaryLight, marginBottom: 16 },

    // Zones
    zone:           { borderRadius: 14, padding: 14, marginBottom: 16 },
    zoneHeader:     { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
    zoneTitle:      { fontSize: 15, fontWeight: '700', color: COLORS.textPrimary },
    zoneCount:      { fontSize: 12, color: COLORS.textSecondary },

    // Item card
    card: {
        backgroundColor: COLORS.surface, padding: 14, borderRadius: 12,
        marginRight: 12, width: 165, elevation: 3,
        shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 6, shadowOffset: { width: 0, height: 2 },
    },
    dirtyCard:      { backgroundColor: COLORS.dirtyBg },
    cardHeader:     { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
    itemCategory:   { fontSize: 14, fontWeight: 'bold', color: COLORS.textPrimary, flex: 1, marginRight: 4 },
    statusBadge:    { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 8 },
    statusText:     { fontSize: 9, fontWeight: '700' },
    cardDetail:     { fontSize: 11, color: COLORS.primaryLight, marginBottom: 8 },
    wearRow:        { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
    wearLabel:      { fontSize: 11, color: COLORS.textMuted },
    wearCount:      { fontSize: 11, fontWeight: '600', color: COLORS.primary },
    scoreRow:       { flexDirection: 'row', alignItems: 'center' },
    scoreLabel:     { fontSize: 10, color: COLORS.textMuted, width: 34 },
    scoreBarBg:     { flex: 1, height: 6, backgroundColor: '#EEEEEE', borderRadius: 3, marginHorizontal: 4 },
    scoreBarFill:   { height: 6, borderRadius: 3 },
    scoreNum:       { fontSize: 10, fontWeight: '700', width: 30, textAlign: 'right' },
    groupBadge:     { marginTop: 6, fontSize: 9, color: COLORS.textMuted },

    // Stats
    statsRow:       { flexDirection: 'row', gap: 10, marginBottom: 16 },
    statCard:       { flex: 1, padding: 14, borderRadius: 12, alignItems: 'center' },
    statNum:        { fontSize: 26, fontWeight: 'bold', color: COLORS.textPrimary },
    statLabel:      { fontSize: 11, color: COLORS.textSecondary, marginTop: 2 },

    // Alert
    alertCard: {
        backgroundColor: '#FFF3E0', borderLeftWidth: 4,
        borderLeftColor: COLORS.accent, padding: 14,
        borderRadius: 10, marginBottom: 16,
    },
    alertTitle:     { fontWeight: 'bold', color: '#E65100', marginBottom: 4 },
    alertBody:      { color: COLORS.textSecondary, fontSize: 13, lineHeight: 20 },

    // Insight section
    insightSection: {
        backgroundColor: COLORS.surface, borderRadius: 12,
        padding: 14, marginBottom: 16, elevation: 2,
        shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4,
    },
    insightSectionTitle: { fontSize: 16, fontWeight: 'bold', color: COLORS.textPrimary, marginBottom: 2 },
    insightSectionSub:   { fontSize: 12, color: COLORS.primaryLight, marginBottom: 10 },
    insightRow: {
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'center', paddingVertical: 8,
        borderTopWidth: 1, borderTopColor: COLORS.divider,
    },
    insightItemName:   { fontSize: 14, fontWeight: '600', color: COLORS.textPrimary },
    insightItemDetail: { fontSize: 12, color: COLORS.primaryLight, marginTop: 1 },
    insightScore:      { fontSize: 17, fontWeight: 'bold' },

    // Laundry
    laundryRow: {
        backgroundColor: COLORS.surface, borderRadius: 12, padding: 14,
        marginBottom: 10, flexDirection: 'row', alignItems: 'center',
        elevation: 2, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4,
    },
    laundryName:   { fontSize: 15, fontWeight: 'bold', color: COLORS.textPrimary },
    laundryDetail: { fontSize: 12, color: COLORS.primaryLight, marginTop: 1 },
    laundryWears:  { fontSize: 11, color: COLORS.textMuted, marginTop: 3 },
    washBtn:       { backgroundColor: COLORS.primary, paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20 },
    washBtnText:   { color: '#FFF', fontSize: 12, fontWeight: '600' },

    // Empty state
    emptyState:    { alignItems: 'center', paddingTop: 60 },
    emptyEmoji:    { fontSize: 52 },
    emptyText:     { fontSize: 16, color: COLORS.primaryLight, marginTop: 12 },

    // Tab bar
    tabBar:        { flexDirection: 'row', backgroundColor: COLORS.surface, borderTopWidth: 1, borderTopColor: COLORS.border, paddingBottom: 6 },
    tabItem:       { flex: 1, alignItems: 'center', paddingTop: 8, position: 'relative' },
    tabIcon:       { fontSize: 20 },
    tabLabel:      { fontSize: 11, color: COLORS.textMuted, marginTop: 2 },
    tabLabelActive: { color: COLORS.primary, fontWeight: '600' },
    tabIndicator:  { position: 'absolute', bottom: 0, width: 32, height: 3, backgroundColor: COLORS.primary, borderRadius: 2 },
});

export default App;
