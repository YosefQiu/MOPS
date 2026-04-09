// cyCodeBase by Cem Yuksel
// [www.cemyuksel.com]
//-------------------------------------------------------------------------------
//! \file   cyColor.h 
//! \author Cem Yuksel
//! 
//! \brief  Color classes.
//!
//-------------------------------------------------------------------------------
//
// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
// All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.
// 
//-------------------------------------------------------------------------------

#ifndef _CY_COLOR_H_INCLUDED_
#define _CY_COLOR_H_INCLUDED_

//-------------------------------------------------------------------------------

#include "cyCore.h"

//-------------------------------------------------------------------------------
namespace cy {
//-------------------------------------------------------------------------------

class Color24;
class Color32;

template <typename T> class Color4;

//-------------------------------------------------------------------------------

//! RGB color class with 3 components

template <typename T>
class Color3
{
	CY_NODISCARD friend Color3<T> operator+( T v, Color3<T> const &c ) { return   c+v;  }	//!< Addition with a constant
	CY_NODISCARD friend Color3<T> operator-( T v, Color3<T> const &c ) { return -(c-v); }	//!< Subtraction from a constant
	CY_NODISCARD friend Color3<T> operator*( T v, Color3<T> const &c ) { return   c*v;  }	//!< Multiplication with a constant

public:

	//!@name Color components
	T r, g, b;

	//!@name Constructors
	Color3() CY_CLASS_FUNCTION_DEFAULT
	explicit Color3( T _r, T _g, T _b ) : r( _r ), g( _g ), b( _b ) {}
	explicit Color3( T const *c )       : r(c[0]), g(c[1]), b(c[2]) {}
	explicit Color3( T rgb )            : r(rgb ), g(rgb ), b(rgb ) {}
	explicit Color3( Color4<T> const &c );
	explicit Color3( Color24   const &c );
	explicit Color3( Color32   const &c );
	template <typename S> Color3( Color3<S> const &c ) : r(c.r), g(c.g), b(c.b) {}
	template <typename S> Color3( Color4<S> const &c ) : r(c.r), g(c.g), b(c.b) {}


	//!@name Set & Get value methods
	void SetBlack()                   { r=0;    g=0;    b=0;    }	//!< Sets r, g and b components as zero.
	void SetWhite()                   { r=1;    g=1;    b=1;    }	//!< Sets r, g and b components as one.
	void Set     ( T _r, T _g, T _b ) { r=_r;   g=_g;   b=_b;   }	//!< Sets r, g and b components as given.
	void Set     ( T const *v )       { r=v[0]; g=v[1]; b=v[2]; }	//!< Sets r, g and b components using the values in the given array.
	void GetValue( T *v )       const { v[0]=r; v[1]=g; v[2]=b; }	//!< Puts r, g and b values into the array.

	//!@name Gray-scale methods
	CY_NODISCARD T Sum  () const { return r + g + b; }
	CY_NODISCARD T Gray () const { return Sum() / T(3); }
	CY_NODISCARD T Luma1() const { return T(0.299 )*r + T(0.587 )*g + T(0.114 )*b; }
	CY_NODISCARD T Luma2() const { return T(0.2126)*r + T(0.7152)*g + T(0.0722)*b; }
	CY_NODISCARD T Min  () const { return r<g ? (r<b ? r : b) : (g<b ? g : b); }
	CY_NODISCARD T Max  () const { return r>g ? (r>b ? r : b) : (g>b ? g : b); }

	//!@name General methods
	CY_NODISCARD bool IsNegative() const { return r< 0 || g< 0 || b< 0; }	//!< Returns true if any component is negative.
	CY_NODISCARD bool IsBlack   () const { return r==0 && g==0 && b==0; }	//!< Returns true if all components are exactly zero.
	CY_NODISCARD bool IsFinite  () const { return cy::IsFinite(r) && cy::IsFinite(g) && cy::IsFinite(b); }	//!< Returns true if all components are finite real numbers.

	CY_NODISCARD Color3<T> Linear2sRGB() const { auto f=[](T cl){ return cl<T(0.0031308) ? cl*T(12.92) : (T)std::pow(cl,T(0.41666))*T(1.055)-T(0.055); }; return Color3<T>(f(r),f(g),f(b)); }	//!< Converts linear RGB to sRGB.
	CY_NODISCARD Color3<T> sRGB2Linear() const { auto f=[](T cs){ return cs<=T(0.04045)  ? cs/T(12.92) : (T)std::pow( (cs+T(0.055))/T(1.055), T(2.4)); }; return Color3<T>(f(r),f(g),f(b)); }	//!< Converts sRGB to linear RGB.

	//!@name Generic template methods
	template <typename F = T(*)(T)>
	void Apply( F func ) { r=func(r); g=func(g); b=func(b); }	//!< Applies the given function to all color components.
	template <typename F = T(*)(T)>
	CY_NODISCARD Color3<T> GetApplied( F func ) const { Color3<T> c=*this; c.Apply<F>(func); return c; }	//!< Returns the resulting color after applying the given function to all color components.

	//!@name Limit methods
	void Clamp   ( T limitMin=0, T limitMax=1 ) { ClampMin(limitMin); ClampMax(limitMax); }
	void ClampMin( T limitMin=0 ) { Apply([&limitMin](T v){ return cy::Max(v,limitMin); }); }
	void ClampMax( T limitMax=1 ) { Apply([&limitMax](T v){ return cy::Min(v,limitMax); }); }
	void Abs() { Apply(std::abs); }

	//!@name Unary operators
	CY_NODISCARD Color3<T>  operator - () const { return Color3<T>(-r,-g,-b); } 

	//!@name Binary operators
	CY_NODISCARD Color3<T> operator + ( Color3<T> const &c ) const { return Color3<T>(r+c.r, g+c.g, b+c.b); }
	CY_NODISCARD Color3<T> operator - ( Color3<T> const &c ) const { return Color3<T>(r-c.r, g-c.g, b-c.b); }
	CY_NODISCARD Color3<T> operator * ( Color3<T> const &c ) const { return Color3<T>(r*c.r, g*c.g, b*c.b); }
	CY_NODISCARD Color3<T> operator / ( Color3<T> const &c ) const { return Color3<T>(r/c.r, g/c.g, b/c.b); }
	CY_NODISCARD Color3<T> operator + ( T                v ) const { return Color3<T>(r+v,   g+v,   b+v  ); }
	CY_NODISCARD Color3<T> operator - ( T                v ) const { return Color3<T>(r-v,   g-v,   b-v  ); }
	CY_NODISCARD Color3<T> operator * ( T                v ) const { return Color3<T>(r*v,   g*v,   b*v  ); }
	CY_NODISCARD Color3<T> operator / ( T                v ) const { return Color3<T>(r/v,   g/v,   b/v  ); }

	//!@name Assignment operators
	Color3<T>& operator += ( Color3<T> const &c ) { r+=c.r; g+=c.g; b+=c.b; return *this; }
	Color3<T>& operator -= ( Color3<T> const &c ) { r-=c.r; g-=c.g; b-=c.b; return *this; }
	Color3<T>& operator *= ( Color3<T> const &c ) { r*=c.r; g*=c.g; b*=c.b; return *this; }
	Color3<T>& operator /= ( Color3<T> const &c ) { r/=c.r; g/=c.g; b/=c.b; return *this; }
	Color3<T>& operator += ( T                v ) { r+=v;   g+=v;   b+=v;   return *this; }
	Color3<T>& operator -= ( T                v ) { r-=v;   g-=v;   b-=v;   return *this; }
	Color3<T>& operator *= ( T                v ) { r*=v;   g*=v;   b*=v;   return *this; }
	Color3<T>& operator /= ( T                v ) { r/=v;   g/=v;   b/=v;   return *this; }

	//!@name Comparison operators
	CY_NODISCARD bool operator == ( Color3<T> const &c ) const { return ( (r==c.r) && (g==c.g) && (b==c.b) ); }
	CY_NODISCARD bool operator != ( Color3<T> const &c ) const { return ( (r!=c.r) || (g!=c.g) || (b!=c.b) ); }
	CY_NODISCARD bool operator <  ( Color3<T> const &c ) const { return ( (r< c.r) && (g< c.g) && (b< c.b) ); }
	CY_NODISCARD bool operator <= ( Color3<T> const &c ) const { return ( (r<=c.r) && (g<=c.g) && (b<=c.b) ); }
	CY_NODISCARD bool operator >  ( Color3<T> const &c ) const { return ( (r> c.r) && (g> c.g) && (b> c.b) ); }
	CY_NODISCARD bool operator >= ( Color3<T> const &c ) const { return ( (r>=c.r) && (g>=c.g) && (b>=c.b) ); }

	//!@name Access operators
	CY_NODISCARD T& operator [] ( int i )       { return (&r)[i]; }
	CY_NODISCARD T  operator [] ( int i ) const { return (&r)[i]; }

	//!@name Static methods
	CY_NODISCARD static Color3<T> Black() { return Color3<T>(0,0,0); }	//!< Returns a black color.
	CY_NODISCARD static Color3<T> White() { return Color3<T>(1,1,1); }	//!< Returns a white color.
};

//-------------------------------------------------------------------------------

typedef Color3<float> Color;

//-------------------------------------------------------------------------------

//! RGBA color class with 4 components

template <typename T>
class Color4
{
	CY_NODISCARD friend Color4<T> operator + ( T v, Color4<T> const &c ) { return   c+v;  }	//!< Addition with a constant
	CY_NODISCARD friend Color4<T> operator - ( T v, Color4<T> const &c ) { return -(c-v); }	//!< Subtraction from a constant
	CY_NODISCARD friend Color4<T> operator * ( T v, Color4<T> const &c ) { return   c*v;  }	//!< Multiplication with a constant

public:

	//!@name Color components
	T r, g, b, a;

	//!@name Constructors
	Color4() CY_CLASS_FUNCTION_DEFAULT
	explicit Color4( T _r, T _g, T _b, T _a=1 )   : r( _r ), g( _g ), b( _b ), a( _a ) {}
	explicit Color4( T const *c )                 : r(c[0]), g(c[1]), b(c[2]), a(c[3]) {}
	explicit Color4( T rgb, T _a=1 )              : r(rgb ), g(rgb ), b(rgb ), a( _a ) {}
	explicit Color4( Color3<T> const &c, T _a=1 ) : r(c.r ), g(c.g ), b(c.b ), a( _a ) {}
	explicit Color4( Color24   const &c, T _a=1 );
	explicit Color4( Color32   const &c );
	template <typename S> explicit Color4( Color3<S> const &c, T _a=1 ) : r(c.r), g(c.g), b(c.b), a( _a) {}
	template <typename S> explicit Color4( Color4<S> const &c )         : r(c.r), g(c.g), b(c.b), a(c.a) {}

	//!@name Set & Get value methods
	void SetBlack( T alpha=1 )                { r=0;    g=0;    b=0;    a=alpha; }	//!< Sets r, g, and b components as zero and a component as given.
	void SetWhite( T alpha=1 )                { r=1;    g=1;    b=1;    a=alpha; }	//!< Sets r, g, and b components as one and a component as given.
	void Set     ( T _r, T _g, T _b, T _a=1 ) { r=_r;   g=_g;   b=_b;   a=_a;    }	//!< Sets r, g, b and a components as given.
	void Set     ( T const *v )               { r=v[0]; g=v[1]; b=v[2]; a=v[3];  }	//!< Sets r, g, b and a components using the values in the given array.
	void GetValue( T *v )               const { v[0]=r; v[1]=g; v[2]=b; v[3]=a;  }	//!< Puts r, g, b and a values into the array.

	//!@name Gray-scale methods
	CY_NODISCARD T Sum  () const { return r + g + b; }
	CY_NODISCARD T Gray () const { return Sum() / T(3); }
	CY_NODISCARD T Luma1() const { return T(0.299 )*r + T(0.587 )*g + T(0.114 )*b; }
	CY_NODISCARD T Luma2() const { return T(0.2126)*r + T(0.7152)*g + T(0.0722)*b; }
	CY_NODISCARD T Min  () const { T mrg = r<g ? r : g; T mba = b<a ? b : a; return mrg<mba ? mrg : mba; }
	CY_NODISCARD T Max  () const { T mrg = r>g ? r : g; T mba = b>a ? b : a; return mrg>mba ? mrg : mba; }

	//!@name General methods
	CY_NODISCARD bool IsNegative() const { return r< 0 || g< 0 || b< 0 || a<0; }	//!< Returns true if any component is negative.
	CY_NODISCARD bool IsBlack   () const { return r==0 && g==0 && b==0; }			//!< Returns true if the r, g, and b components are exactly zero.
	CY_NODISCARD bool IsFinite  () const { return cy::IsFinite(r) && cy::IsFinite(g) && cy::IsFinite(b) && cy::IsFinite(a); }	//!< Returns true if all components are finite real numbers.

	CY_NODISCARD Color4<T> Linear2sRGB() const { return Color4<T>(Color(r,g,b).Linear2sRGB(),a); }	//!< Converts linear RGB to sRGB.
	CY_NODISCARD Color4<T> sRGB2Linear() const { return Color4<T>(Color(r,g,b).sRGB2Linear(),a); }	//!< Converts sRGB to linear RGB.

	//!@name Generic template methods
	template <typename F = T(*)(T)>
	void Apply( F func ) { r=func(r); g=func(g); b=func(b); }	//!< Applies the given function to all color components.
	template <typename F = T(*)(T)>
	CY_NODISCARD Color4<T> GetApplied( F func ) const { Color4<T> c=*this; c.Apply<F>(func); return c; }	//!< Returns the resulting color after applying the given function to all color components.

	//!@name Limit methods
	void Clamp   ( T limitMin=0, T limitMax=1 ) { ClampMin(limitMin); ClampMax(limitMax); }
	void ClampMin( T limitMin=0 ) { Apply([&limitMin](T v){ return cy::Max(v,limitMin); }); }
	void ClampMax( T limitMax=1 ) { Apply([&limitMax](T v){ return cy::Min(v,limitMax); }); }
	void Abs() { Apply(std::abs); }

	//!@name Unary operators
	CY_NODISCARD Color4<T>  operator - () const { return Color4<T>(-r,-g,-b,-a); } 

	//!@name Binary operators
	CY_NODISCARD Color4<T> operator + ( Color4<T> const &c ) const { return Color4<T>(r+c.r, g+c.g, b+c.b, a+c.a); }
	CY_NODISCARD Color4<T> operator - ( Color4<T> const &c ) const { return Color4<T>(r-c.r, g-c.g, b-c.b, a-c.a); }
	CY_NODISCARD Color4<T> operator * ( Color4<T> const &c ) const { return Color4<T>(r*c.r, g*c.g, b*c.b, a*c.a); }
	CY_NODISCARD Color4<T> operator / ( Color4<T> const &c ) const { return Color4<T>(r/c.r, g/c.g, b/c.b, a/c.a); }
	CY_NODISCARD Color4<T> operator + ( T                v ) const { return Color4<T>(r+v,   g+v,   b+v,   a+v  ); }
	CY_NODISCARD Color4<T> operator - ( T                v ) const { return Color4<T>(r-v,   g-v,   b-v,   a-v  ); }
	CY_NODISCARD Color4<T> operator * ( T                v ) const { return Color4<T>(r*v,   g*v,   b*v,   a*v  ); }
	CY_NODISCARD Color4<T> operator / ( T                v ) const { return Color4<T>(r/v,   g/v,   b/v,   a/v  ); }

	//!@name Assignment operators
	Color4<T>& operator += ( Color4<T> const &c ) { r+=c.r; g+=c.g; b+=c.b; a+=c.a; return *this; }
	Color4<T>& operator -= ( Color4<T> const &c ) { r-=c.r; g-=c.g; b-=c.b; a-=c.a; return *this; }
	Color4<T>& operator *= ( Color4<T> const &c ) { r*=c.r; g*=c.g; b*=c.b; a*=c.a; return *this; }
	Color4<T>& operator /= ( Color4<T> const &c ) { r/=c.r; g/=c.g; b/=c.b; a/=c.a; return *this; }
	Color4<T>& operator += ( T                v ) { r+=v;   g+=v;   b+=v;   a+=v;   return *this; }
	Color4<T>& operator -= ( T                v ) { r-=v;   g-=v;   b-=v;   a-=v;   return *this; }
	Color4<T>& operator *= ( T                v ) { r*=v;   g*=v;   b*=v;   a*=v;   return *this; }
	Color4<T>& operator /= ( T                v ) { r/=v;   g/=v;   b/=v;   a/=v;   return *this; }

	//!@name Comparison operators
	CY_NODISCARD bool operator == ( Color4<T> const &c ) const { return ( (r==c.r) && (g==c.g) && (b==c.b) && (a==c.a) ); }
	CY_NODISCARD bool operator != ( Color4<T> const &c ) const { return ( (r!=c.r) || (g!=c.g) || (b!=c.b) || (a!=c.a) ); }
	CY_NODISCARD bool operator <  ( Color4<T> const &c ) const { return ( (r< c.r) && (g< c.g) && (b< c.b) && (a< c.a) ); }
	CY_NODISCARD bool operator <= ( Color4<T> const &c ) const { return ( (r<=c.r) && (g<=c.g) && (b<=c.b) && (a<=c.a) ); }
	CY_NODISCARD bool operator >  ( Color4<T> const &c ) const { return ( (r> c.r) && (g> c.g) && (b> c.b) && (a> c.a) ); }
	CY_NODISCARD bool operator >= ( Color4<T> const &c ) const { return ( (r>=c.r) && (g>=c.g) && (b>=c.b) && (a>=c.a) ); }

	//!@name Access operators
	CY_NODISCARD T& operator [] ( int i )       { return (&r)[i]; }
	CY_NODISCARD T  operator [] ( int i ) const { return (&r)[i]; }

	//!@name Static methods
	CY_NODISCARD static Color4<T> Black( T alpha=1 ) { return Color4<T>(0,0,0,alpha); }	//!< Returns a black color.
	CY_NODISCARD static Color4<T> White( T alpha=1 ) { return Color4<T>(1,1,1,alpha); }	//!< Returns a white color.
};

//-------------------------------------------------------------------------------

typedef Color4<float> ColorA;

//-------------------------------------------------------------------------------

//! 24-bit RGB color class with 3 unsigned byte components

class Color24
{
public:

	//!@name Color components
	uint8_t r, g, b;

	//!@name Constructors
	Color24() CY_CLASS_FUNCTION_DEFAULT
	Color24( Color24 const &c ) : r(c.r), g(c.g), b(c.b) {}
	explicit Color24( uint8_t _r, uint8_t _g, uint8_t _b ) : r(_r), g(_g), b(_b) {}
	explicit Color24( Color32 const &c );
	template <typename T> explicit Color24( Color3<T> const &c ) { r=ToByte(c.r); g=ToByte(c.g); b=ToByte(c.b); }
	template <typename T> explicit Color24( Color4<T> const &c ) { r=ToByte(c.r); g=ToByte(c.g); b=ToByte(c.b); }

	//!@name Conversion methods
	CY_NODISCARD Color  ToColor () const { return Color (r/255.0f,g/255.0f,b/255.0f); }
	CY_NODISCARD ColorA ToColorA() const { return ColorA(r/255.0f,g/255.0f,b/255.0f,1); }

	//!@name Set & Get value methods
	void SetBlack() { r=  0; g=  0; b=  0; }									//!< Sets r, g, and b components as zero.
	void SetWhite() { r=255; g=255; b=255; }									//!< Sets r, g, and b components as 255.
	void Set     ( uint8_t _r, uint8_t _g, uint8_t _b ) { r=_r; g=_g; b=_b; }	//!< Sets r, g, and b components as given.
	void Set     ( uint8_t const *v ) { r=v[0]; g=v[1]; b=v[2]; }				//!< Sets r, g, and b components using the values in the given array.
	void GetValue( uint8_t *v ) const { v[0]=r; v[1]=g; v[2]=b; }				//!< Puts r, g, and b values into the array.

	//!@name Gray-scale methods
	CY_NODISCARD int     Sum () const { return int(r) + int(g) + int(b); }
	CY_NODISCARD uint8_t Gray() const { return uint8_t( (Sum()+1) / 3 ); }
	CY_NODISCARD uint8_t Min () const { return r<g ? (r<b ? r : b) : (g<b ? g : b); }
	CY_NODISCARD uint8_t Max () const { return r>g ? (r>b ? r : b) : (g>b ? g : b); }

	//!@name General methods
	CY_NODISCARD bool IsBlack() const { return r==0 && g==0 && b==0; }	//!< Returns true if all components are exactly zero.

	//!@name Limit methods
	void Clamp   ( uint8_t limitMin=  0, uint8_t limitMax=255 ) { ClampMin(limitMin); ClampMax(limitMax); }
	void ClampMin( uint8_t limitMin=  0 ) { r=cy::Max(r,limitMin); g=cy::Max(g,limitMin); b=cy::Max(b,limitMin); }
	void ClampMax( uint8_t limitMax=255 ) { r=cy::Min(r,limitMax); g=cy::Min(g,limitMax); b=cy::Min(b,limitMax); }

	//!@name Comparison operators
	CY_NODISCARD bool operator == ( Color24 const &c ) const { return ( (r==c.r) && (g==c.g) && (b==c.b) ); }
	CY_NODISCARD bool operator != ( Color24 const &c ) const { return ( (r!=c.r) || (g!=c.g) || (b!=c.b) ); }
	CY_NODISCARD bool operator <  ( Color24 const &c ) const { return ( (r< c.r) && (g< c.g) && (b< c.b) ); }
	CY_NODISCARD bool operator <= ( Color24 const &c ) const { return ( (r<=c.r) && (g<=c.g) && (b<=c.b) ); }
	CY_NODISCARD bool operator >  ( Color24 const &c ) const { return ( (r> c.r) && (g> c.g) && (b> c.b) ); }
	CY_NODISCARD bool operator >= ( Color24 const &c ) const { return ( (r>=c.r) && (g>=c.g) && (b>=c.b) ); }

	//!@name Access operators
	CY_NODISCARD uint8_t& operator [] ( int i )       { return (&r)[i]; }
	CY_NODISCARD uint8_t  operator [] ( int i ) const { return (&r)[i]; }

	//!@name Static methods
	CY_NODISCARD static Color24 Black() { return Color24(  0,  0,  0); }	//!< Returns a black color.
	CY_NODISCARD static Color24 White() { return Color24(255,255,255); }	//!< Returns a white color.

protected:
	template <typename T> CY_NODISCARD static uint8_t ToByte (T r) { return ClampInt(int(r*255+T(0.5))); }
	CY_NODISCARD static uint8_t ClampInt(int v) { return v<0 ? 0 : (v>255 ? 255 : static_cast<uint8_t>(v)); }
};

//-------------------------------------------------------------------------------

//! 32-bit RGBA color class with 4 unsigned byte components

class Color32
{
public:

	//!@name Color components
	uint8_t r, g, b, a;

	//!@name Constructors
	Color32() CY_CLASS_FUNCTION_DEFAULT
	Color32( Color32 const &c ) : r(c.r), g(c.g), b(c.b), a(c.a) {}
	explicit Color32( uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a=255 ) : r(_r), g(_g), b(_b), a(_a) {}
	explicit Color32( Color24 const &c, uint8_t _a=255 ) : r(c.r), g(c.g), b(c.b), a(_a) {}
	template <typename T> explicit Color32( Color3<T> const &c, T _a=1 ) { r=ToByte(c.r); g=ToByte(c.g); b=ToByte(c.b); }
	template <typename T> explicit Color32( Color4<T> const &c )         { r=ToByte(c.r); g=ToByte(c.g); b=ToByte(c.b); }

	//!@name Conversion methods
	CY_NODISCARD Color  ToColor () const { return Color (r/255.0f,g/255.0f,b/255.0f); }
	CY_NODISCARD ColorA ToColorA() const { return ColorA(r/255.0f,g/255.0f,b/255.0f,a/255.0f); }

	//!@name Set & Get value methods
	void SetBlack( uint8_t _a=255 ) { r=  0; g=  0; b=  0; a=_a; }								//!< Sets r, g, and b components as zero and a component as given.
	void SetWhite( uint8_t _a=255 ) { r=255; g=255; b=255; a=_a; }								//!< Sets r, g, and b components as 255 and a component as given.
	void Set     ( uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a ) { r=_r; g=_g; b=_b; a=_a; }	//!< Sets r, g, b and a components as given.
	void Set     ( uint8_t const *v ) { r=v[0]; g=v[1]; b=v[2]; a=v[3]; }						//!< Sets r, g, b and a components using the values in the given array.
	void GetValue( uint8_t *v ) const { v[0]=r; v[1]=g; v[2]=b; v[3]=a; }						//!< Puts r, g, b and a values into the array.

	//!@name Gray-scale methods
	CY_NODISCARD int     Sum () const { return int(r) + int(g) + int(b); }
	CY_NODISCARD uint8_t Gray() const { return uint8_t( (Sum()+1) / 3 ); }
	CY_NODISCARD uint8_t Min () const { uint8_t mrg = r<g ? r : g; uint8_t mba = b<a ? b : a; return mrg<mba ? mrg : mba; }
	CY_NODISCARD uint8_t Max () const { uint8_t mrg = r>g ? r : g; uint8_t mba = b>a ? b : a; return mrg>mba ? mrg : mba; }

	//!@name General methods
	CY_NODISCARD bool IsBlack() const { return r==0 && g==0 && b==0; }	//!< Returns true if the r, g, and b components are exactly zero.

	//!@name Limit methods
	void Clamp   ( uint8_t limitMin=  0, uint8_t limitMax=255 ) { ClampMin(limitMin); ClampMax(limitMax); }
	void ClampMin( uint8_t limitMin=  0 ) { r=cy::Max(r,limitMin); g=cy::Max(g,limitMin); b=cy::Max(b,limitMin); a=cy::Max(a,limitMin); }
	void ClampMax( uint8_t limitMax=255 ) { r=cy::Min(r,limitMax); g=cy::Min(g,limitMax); b=cy::Min(b,limitMax); a=cy::Min(a,limitMax); }

	//!@name Comparison operators
	CY_NODISCARD bool operator == ( Color32 const &c ) const { return ( (r==c.r) && (g==c.g) && (b==c.b) && (a==c.a) ); }
	CY_NODISCARD bool operator != ( Color32 const &c ) const { return ( (r!=c.r) || (g!=c.g) || (b!=c.b) || (a!=c.a) ); }
	CY_NODISCARD bool operator <  ( Color32 const &c ) const { return ( (r< c.r) && (g< c.g) && (b< c.b) && (a< c.a) ); }
	CY_NODISCARD bool operator <= ( Color32 const &c ) const { return ( (r<=c.r) && (g<=c.g) && (b<=c.b) && (a<=c.a) ); }
	CY_NODISCARD bool operator >  ( Color32 const &c ) const { return ( (r> c.r) && (g> c.g) && (b> c.b) && (a> c.a) ); }
	CY_NODISCARD bool operator >= ( Color32 const &c ) const { return ( (r>=c.r) && (g>=c.g) && (b>=c.b) && (a>=c.a) ); }

	//!@name Access operators
	CY_NODISCARD uint8_t& operator [] ( int i )       { return (&r)[i]; }
	CY_NODISCARD uint8_t  operator [] ( int i ) const { return (&r)[i]; }

	//!@name Static methods
	CY_NODISCARD static Color32 Black( uint8_t alpha=255 ) { return Color32(  0,  0,  0,alpha); }	//!< Returns a black color.
	CY_NODISCARD static Color32 White( uint8_t alpha=255 ) { return Color32(255,255,255,alpha); }	//!< Returns a white color.

protected:
	template <typename T> CY_NODISCARD static uint8_t ToByte (T r) { return ClampInt(int(r*255+T(0.5))); }
	CY_NODISCARD static uint8_t ClampInt(int v) { return v<0 ? 0 : (v>255 ? 255 : static_cast<uint8_t>(v)); }
};

//-------------------------------------------------------------------------------

//!@name Common math functions

template <typename T> inline Color3<T> Abs  ( Color3<T> const &c ) { return c.GetApplied(std::abs   ); }	//!< Returns a color with abs applied to all components.
template <typename T> inline Color3<T> Exp  ( Color3<T> const &c ) { return c.GetApplied(std::exp   ); }	//!< Returns a color with exponential applied to all components.
template <typename T> inline Color3<T> Exp2 ( Color3<T> const &c ) { return c.GetApplied(std::exp2  ); }	//!< Returns a color with base-2 exponential applied to all components.
template <typename T> inline Color3<T> Log  ( Color3<T> const &c ) { return c.GetApplied(std::log   ); }	//!< Returns a color with natural logarithm applied to all components.
template <typename T> inline Color3<T> Log2 ( Color3<T> const &c ) { return c.GetApplied(std::log2  ); }	//!< Returns a color with base-2 logarithm applied to all components.
template <typename T> inline Color3<T> Log10( Color3<T> const &c ) { return c.GetApplied(std::log10 ); }	//!< Returns a color with base-10 logarithm applied to all components.
template <typename T> inline Color3<T> Sqrt ( Color3<T> const &c ) { return c.GetApplied(cy::Sqrt<T>); }	//!< Returns a color with square root applied to all components.
template <typename T> inline Color3<T> Pow  ( Color3<T> const &c, T exponent ) { return c.GetApplied([&exponent](T v){ return (T)std::pow(v,exponent); }); }	//!< Returns a color with square root applied to all components.

template <typename T> inline Color4<T> Abs  ( Color4<T> const &c ) { return c.GetApplied(std::abs   ); }	//!< Returns a color with abs applied to all components.
template <typename T> inline Color4<T> Exp  ( Color4<T> const &c ) { return c.GetApplied(std::exp   ); }	//!< Returns a color with exponential applied to all components.
template <typename T> inline Color4<T> Exp2 ( Color4<T> const &c ) { return c.GetApplied(std::exp2  ); }	//!< Returns a color with base-2 exponential applied to all components.
template <typename T> inline Color4<T> Log  ( Color4<T> const &c ) { return c.GetApplied(std::log   ); }	//!< Returns a color with natural logarithm applied to all components.
template <typename T> inline Color4<T> Log2 ( Color4<T> const &c ) { return c.GetApplied(std::log2  ); }	//!< Returns a color with base-2 logarithm applied to all components.
template <typename T> inline Color4<T> Log10( Color4<T> const &c ) { return c.GetApplied(std::log10 ); }	//!< Returns a color with base-10 logarithm applied to all components.
template <typename T> inline Color4<T> Sqrt ( Color4<T> const &c ) { return c.GetApplied(cy::Sqrt<T>); }	//!< Returns a color with square root applied to all components.
template <typename T> inline Color4<T> Pow  ( Color4<T> const &c, T exponent ) { return c.GetApplied([&exponent](T v){ return (T)std::pow(v,exponent); }); }	//!< Returns a color with square root applied to all components.

//-------------------------------------------------------------------------------

template <typename T> inline Color3<T>::Color3( Color4<T> const &c ) : r(c.r), g(c.g), b(c.b) {}
template <typename T> inline Color3<T>::Color3( Color24   const &c ) : Color3<T>( c.ToColor() ) {}
template <typename T> inline Color3<T>::Color3( Color32   const &c ) : Color3<T>( c.ToColor() ) {}
template <typename T> inline Color4<T>::Color4( Color24   const &c, T alpha ) { Color3<T> rgb=c.ToColor(); r=rgb.r; g=rgb.g; b=rgb.b; a=alpha; }
template <typename T> inline Color4<T>::Color4( Color32   const &c ) : Color4<T>( c.ToColorA() ) {}
inline Color24::Color24( Color32 const &c ) : r(c.r), g(c.g), b(c.b) {}

//-------------------------------------------------------------------------------

typedef Color3<float>  Color3f;	//!< RGB color class with 3 float components
typedef Color4<float>  Color4f;	//!< RGBA color class with 4 float components

typedef Color3<double> Color3d;	//!< RGB color class with 3 double components
typedef Color4<double> Color4d;	//!< RGBA color class with 4 double components

//-------------------------------------------------------------------------------
} // namespace cy
//-------------------------------------------------------------------------------

typedef cy::Color   cyColor;	//!< RGB color class with 3 float components
typedef cy::Color3f cyColor3f;	//!< RGB color class with 3 float components
typedef cy::Color3d cyColor3d;	//!< RGB color class with 3 double components
typedef cy::ColorA  cyColorA;	//!< RGBA color class with 4 float components
typedef cy::Color4f cyColor4f;	//!< RGBA color class with 4 float components
typedef cy::Color4d cyColor4d;	//!< RGBA color class with 4 double components
typedef cy::Color24 cyColor24;	//!< 24-bit RGB color class with 3 unsigned byte components
typedef cy::Color32 cyColor32;	//!< 32-bit RGBA color class with 4 unsigned byte components

//-------------------------------------------------------------------------------

#endif

