//+------------------------------------------------------------------+
//|                                                XAU D1 BO Trail  |
//|                                Daily Breakout/Breakdown EA (MT5) |
//|                                 Author: ChatGPT for Pushkar Singh |
//|                              Description:                        |
//|  - Places 2 pending orders each new server day on XAUUSD:        |
//|      * Buy Stop @ previous day's High                            |
//|      * Sell Stop @ previous day's Low                            |
//|  - Unfilled orders are canceled at start of the next day.        |
//|  - If one side triggers, the other pending remains and may fill. |
//|  - Initial SL/TP = 100 pips (pip = $0.10)                        |
//|  - Trailing: when +10 pips profit, lock SL to +4 pips, then      |
//|    trail by 10 pips with a 10-pip step.                          |
//|  - Fixed lot size = 0.10 by default.                             |
//|                                                                  |
//|  NOTE: "pip" here is defined as $0.10 in price for XAUUSD.       |
//|        100 pips = $10.                                           |
//+------------------------------------------------------------------+
#property copyright   "ChatGPT for Pushkar Singh"
#property version     "1.00"
#property description "Daily D1 breakout EA for XAUUSD with lock & trailing"
#property strict

#include <Trade/Trade.mqh>

//============================== Inputs ==============================
input string   InpTradeSymbol          = "";        // Leave blank to trade the chart symbol (recommended)
input double   InpLots                 = 0.10;      // Fixed lot size per order
input double   InpPipSize              = 0.10;      // Pip size in price units (XAUUSD: 0.10 = $0.10)
input int      InpInitialSL_Pips       = 100;       // Initial SL (pips)
input int      InpInitialTP_Pips       = 100;       // Initial TP (pips)
input int      InpTrailActivate_Pips   = 10;        // Activate trailing at +10 pips
input int      InpTrailLock_Pips       = 4;         // Lock SL to +4 pips on activation
input int      InpTrailOffset_Pips     = 10;        // Trail distance = 10 pips
input int      InpTrailStep_Pips       = 10;        // Modify SL only if improved by >= 10 pips
input bool     InpUseBufferOnStops     = false;     // Add buffer to breakout levels?
input int      InpStopBuffer_Pips      = 0;         // Buffer (pips) if above is true
input uint     InpMagicNumber          = 20250904;  // Magic number for this EA
input int      InpDeviationPoints      = 20;        // Max deviation in points (modifications)
input bool     InpVerboseLogs          = true;      // Extra logging

//============================= Globals =============================
CTrade           trade;
string           g_symbol;
int              g_digits = 0;
double           g_point  = 0.0;
double           g_ticksize = 0.0;
datetime         g_lastD1BarTime = 0;  // Tracks current D1 bar start ("today")

//---------------- Utility: Logging ---------------------------------
void Log(string msg)
{
   if(InpVerboseLogs) Print("[XAU-D1-BO] ", msg);
}

void LogErr(string ctx, int code)
{
   Print("[XAU-D1-BO][ERROR] ", ctx, " | retcode=", code);
}

//---------------- Utility: Symbol/Price helpers --------------------
double PipToPrice(const double pips) { return pips * InpPipSize; }
double PriceToPips(const double price_diff) { return price_diff / InpPipSize; }

double NormalizePrice(double price)
{
   // Round to the instrument's trade tick size if available, otherwise to digits
   double ts = g_ticksize > 0.0 ? g_ticksize : g_point;
   if(ts <= 0.0) return NormalizeDouble(price, g_digits);
   return MathRound(price/ts) * ts;
}

bool GetTicks(MqlTick &tick)
{
   if(!SymbolInfoTick(g_symbol, tick))
   {
      Log("SymbolInfoTick failed");
      return false;
   }
   return true;
}

//---------------- Utility: Broker constraints ----------------------
// Returns minimal distance in price units required for pending orders from current price
double MinDistancePrice()
{
   long stops_level_pts = 0;
   if(!SymbolInfoInteger(g_symbol, SYMBOL_TRADE_STOPS_LEVEL, stops_level_pts))
      stops_level_pts = 0;
   return (double)stops_level_pts * g_point;
}

//---------------- Utility: D1 previous High/Low --------------------
bool GetPrevDayHighLow(double &prevHigh, double &prevLow)
{
   // We need at least 2 D1 bars: [0]=today, [1]=yesterday
   MqlRates rates[];
   int copied = CopyRates(g_symbol, PERIOD_D1, 0, 3, rates);
   if(copied < 2)
   {
      Log("CopyRates D1 < 2; cannot fetch previous day OHLC");
      return false;
   }
   // rates[0] is current day, rates[1] is previous day
   prevHigh = rates[1].high;
   prevLow  = rates[1].low;
   return (prevHigh > 0 && prevLow > 0 && prevHigh >= prevLow);
}

//---------------- Utility: Day change detection --------------------
bool NewDayDetected()
{
   datetime curD1 = iTime(g_symbol, PERIOD_D1, 0);
   if(curD1 <= 0) return false;
   if(curD1 != g_lastD1BarTime)
   {
      g_lastD1BarTime = curD1;
      return true;
   }
   return false;
}

//---------------- Orders & Positions: Filters ----------------------
bool IsOurMagic(const long mg) { return (mg == (long)InpMagicNumber); }
bool IsOurSymbol(const string s) { return (StringCompare(s, g_symbol) == 0); }

// Count our pending orders on this symbol
int CountOurPendings()
{
   int cnt = 0;
   int total = OrdersTotal();
   for(int i=0; i<total; ++i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if(!IsOurSymbol(OrderGetString(ORDER_SYMBOL))) continue;
      if(!IsOurMagic((long)OrderGetInteger(ORDER_MAGIC))) continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_SELL_STOP)
         ++cnt;
   }
   return cnt;
}

// Delete all our pending stop orders on this symbol
void DeleteOurPendings()
{
   int total = OrdersTotal();
   for(int i = total-1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if(!IsOurSymbol(OrderGetString(ORDER_SYMBOL))) continue;
      if(!IsOurMagic((long)OrderGetInteger(ORDER_MAGIC))) continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_SELL_STOP)
      {
         if(!trade.OrderDelete(ticket))
            LogErr("OrderDelete(" + (string)ticket + ")", (int)trade.ResultRetcode());
         else
            Log("Deleted pending order ticket=" + (string)ticket);
      }
   }
}

// Modify SL/TP for a specific position ticket
bool ModifyPositionSLTP(const ulong pos_ticket, const double newSL, const double newTP)
{
   MqlTradeRequest  req;
   MqlTradeResult   res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action   = TRADE_ACTION_SLTP;
   req.position = pos_ticket;
   req.symbol   = g_symbol;
   req.sl       = (newSL > 0.0 ? NormalizePrice(newSL) : 0.0);
   req.tp       = (newTP > 0.0 ? NormalizePrice(newTP) : 0.0);
   req.magic    = (long)InpMagicNumber;

   if(!OrderSend(req, res))
   {
      Log("OrderSend SLTP failed; ticket=" + (string)pos_ticket + ", retcode=" + (string)res.retcode);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE)
   {
      Log("SLTP modify retcode != DONE; ticket=" + (string)pos_ticket + ", retcode=" + (string)res.retcode);
      return false;
   }
   return true;
}

//---------------- Placement: Daily Buy Stop & Sell Stop -------------
bool PlaceDailyPendings()
{
   double prevHigh, prevLow;
   if(!GetPrevDayHighLow(prevHigh, prevLow)) return false;

   if(prevHigh <= 0 || prevLow <= 0 || prevHigh < prevLow)
   {
      Log("Invalid prev day High/Low; skipping placement");
      return false;
   }

   MqlTick tick;
   if(!GetTicks(tick)) return false;

   double minDist = MinDistancePrice();

   // Apply optional buffer
   double buffer = InpUseBufferOnStops ? PipToPrice((double)InpStopBuffer_Pips) : 0.0;

   //--- Buy Stop @ prevHigh (+buffer) (must be > Ask + minDist)
   double buyStopPrice = prevHigh + buffer;
   double minBuy = tick.ask + minDist;
   if(buyStopPrice < minBuy) buyStopPrice = minBuy;
   buyStopPrice = NormalizePrice(buyStopPrice);

   double buySL = NormalizePrice(buyStopPrice - PipToPrice((double)InpInitialSL_Pips));
   double buyTP = NormalizePrice(buyStopPrice + PipToPrice((double)InpInitialTP_Pips));

   //--- Sell Stop @ prevLow (-buffer) (must be < Bid - minDist)
   double sellStopPrice = prevLow - buffer;
   double maxSell = tick.bid - minDist;
   if(sellStopPrice > maxSell) sellStopPrice = maxSell;
   sellStopPrice = NormalizePrice(sellStopPrice);

   double sellSL = NormalizePrice(sellStopPrice + PipToPrice((double)InpInitialSL_Pips));
   double sellTP = NormalizePrice(sellStopPrice - PipToPrice((double)InpInitialTP_Pips));

   // Place orders
   trade.SetExpertMagicNumber((long)InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   string commentTag = StringFormat("XAU_D1_BO %s", TimeToString(g_lastD1BarTime, TIME_DATE));

   bool ok1 = trade.BuyStop(InpLots, buyStopPrice, g_symbol, buySL, buyTP, ORDER_TIME_GTC, 0, commentTag);
   if(!ok1) LogErr("BuyStop @" + DoubleToString(buyStopPrice, g_digits), (int)trade.ResultRetcode()); else Log("Placed BuyStop @" + DoubleToString(buyStopPrice, g_digits));

   bool ok2 = trade.SellStop(InpLots, sellStopPrice, g_symbol, sellSL, sellTP, ORDER_TIME_GTC, 0, commentTag);
   if(!ok2) LogErr("SellStop @" + DoubleToString(sellStopPrice, g_digits), (int)trade.ResultRetcode()); else Log("Placed SellStop @" + DoubleToString(sellStopPrice, g_digits));

   return (ok1 || ok2);
}

//---------------- Trailing stop management -------------------------
void ManageTrailing()
{
   MqlTick tick;
   if(!GetTicks(tick)) return;

   int total = PositionsTotal();
   for(int i=0; i<total; ++i)
   {
     ulong pos_ticket = PositionGetTicket(i);
     if(pos_ticket == 0) continue;
     if(!PositionSelectByTicket(pos_ticket)) continue;

     string sym      = PositionGetString(POSITION_SYMBOL);
     if(!IsOurSymbol(sym)) continue;
     long   magic    = PositionGetInteger(POSITION_MAGIC);
     if(!IsOurMagic(magic)) continue;

     long   type     = PositionGetInteger(POSITION_TYPE);
     double entry    = PositionGetDouble(POSITION_PRICE_OPEN);
     double curSL    = PositionGetDouble(POSITION_SL);
     double curTP    = PositionGetDouble(POSITION_TP);

     // Compute profit in pips per your definition (use Bid for long, Ask for short)
     double profit_pips = 0.0;
     if(type == POSITION_TYPE_BUY)
        profit_pips = PriceToPips(tick.bid - entry);
     else if(type == POSITION_TYPE_SELL)
        profit_pips = PriceToPips(entry - tick.ask);
     else
        continue;

     // Activation check
     if(profit_pips >= (double)InpTrailActivate_Pips)
     {
        double lockSL = 0.0;
        if(type == POSITION_TYPE_BUY)
           lockSL = entry + PipToPrice((double)InpTrailLock_Pips);
        else // sell
           lockSL = entry - PipToPrice((double)InpTrailLock_Pips);

        // If SL not yet locked to at least the lock level, do it first
        bool needLock = false;
        if(curSL <= 0.0)
        {
           needLock = true;
        }
        else
        {
           if(type == POSITION_TYPE_BUY && curSL < lockSL - (g_point*0.5)) needLock = true;
           if(type == POSITION_TYPE_SELL && curSL > lockSL + (g_point*0.5)) needLock = true;
        }

        if(needLock)
        {
           double newSL = NormalizePrice(lockSL);
           if(ModifyPositionSLTP(pos_ticket, newSL, curTP))
              Log(StringFormat("Ticket %I64u: Lock SL to %s", pos_ticket, DoubleToString(newSL, g_digits)));
           else
              LogErr("Lock SL modify", (int)trade.ResultRetcode());
           // Update curSL for subsequent trailing calc
           curSL = PositionGetDouble(POSITION_SL);
        }

        // Now compute trailing desired SL
        double desiredSL = 0.0;
        if(type == POSITION_TYPE_BUY)
        {
           desiredSL = tick.bid - PipToPrice((double)InpTrailOffset_Pips);
           // Never below lock level
           double minSL = entry + PipToPrice((double)InpTrailLock_Pips);
           if(desiredSL < minSL) desiredSL = minSL;

           // Update only if improves by at least the step
           if(curSL <= 0.0 || desiredSL > curSL + PipToPrice((double)InpTrailStep_Pips))
           {
              double newSL = NormalizePrice(desiredSL);
              if(ModifyPositionSLTP(pos_ticket, newSL, curTP))
                 Log(StringFormat("Ticket %I64u: Trail SL to %s", pos_ticket, DoubleToString(newSL, g_digits)));
              else
                 LogErr("Trail SL modify", (int)trade.ResultRetcode());
           }
        }
        else if(type == POSITION_TYPE_SELL)
        {
           desiredSL = tick.ask + PipToPrice((double)InpTrailOffset_Pips);
           // Never above lock level
           double maxSL = entry - PipToPrice((double)InpTrailLock_Pips);
           if(desiredSL > maxSL) desiredSL = maxSL;

           if(curSL <= 0.0 || desiredSL < curSL - PipToPrice((double)InpTrailStep_Pips))
           {
              double newSL = NormalizePrice(desiredSL);
              if(ModifyPositionSLTP(pos_ticket, newSL, curTP))
                 Log(StringFormat("Ticket %I64u: Trail SL to %s", pos_ticket, DoubleToString(newSL, g_digits)));
              else
                 LogErr("Trail SL modify", (int)trade.ResultRetcode());
           }
        }
     }
   }
}

//---------------- Daily reset (once per new day) -------------------
void DailyReset()
{
   Log("New server day detected -> cancel old pendings and place new ones");
   DeleteOurPendings();
   PlaceDailyPendings();
}

//============================= MT5 Events ===========================
int OnInit()
{
   g_symbol = (InpTradeSymbol == "" ? _Symbol : InpTradeSymbol);

   // Load symbol properties
   g_digits   = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   g_point    = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   g_ticksize = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);

   if(!SymbolSelect(g_symbol, true))
   {
      Print("[XAU-D1-BO][FATAL] Cannot select symbol ", g_symbol);
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber((long)InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   // Initialize current D1 bar time for day detection
   g_lastD1BarTime = iTime(g_symbol, PERIOD_D1, 0);

   // If there are no our pendings yet today, place them once at init
   if(CountOurPendings() == 0)
   {
      Log("OnInit: No existing pendings found -> placing today's orders");
      // Do not delete anything at init; respect existing trader placements if any
      PlaceDailyPendings();
   }
   else
   {
      Log("OnInit: Found existing pendings -> leaving them until daily reset");
   }

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   // No special cleanup
}

void OnTick()
{
   // Handle new day boundary first
   if(NewDayDetected())
   {
      DailyReset();
   }

   // Manage trailing stops for all our open positions
   ManageTrailing();
}

//============================= END =================================
