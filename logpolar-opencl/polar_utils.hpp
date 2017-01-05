#ifndef POLAR_UTILS_HPP
#define POLAR_UTILS_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#define MAX_PIX_COUNT (32)

#define PI (3.14159265359)


//Autor: Piotr Konopka
//Nazwa: linspace
//Przeznaczenie: Funkcja pomocniczna dla create_maps()
//Opis dzia³ania: Funkcja tworzy wektor równoogleglych liczb na zadanym przedziale.
//Argumenty:
// - float y0 - pierwsza wartosc wektora
// - float ymax - ostatnia wartosc wektora
// - float steps - liczba elementów wektora
// - std::vector<float> & vec - referencja zwracanego wektora
//Wartoœæ zwracana: void
//Wykorzystywane biblioteki: STL
void linspace( float y0, float ymax, float steps, std::vector<float> & vec);



//Autor: Piotr Konopka, Micha³ Góra (model matlab)
//Nazwa: create_maps
//Przeznaczenie: Funkcja wykorzystywana na pocz¹tku programu do utworzenia map
// transformacji.
//Opis dzia³ania: Funkcja tworzy mapy transformacji przechodz¹c po kolejnych
// koordynatach obrazu polarnego.
//Argumenty:
// - cv::Mat & to_polar_map - referencja zwracanej macierzy mapy transformacji
//    polarnej
// - cv::Mat & to_cart_map - referencja zwracanej macierzy mapy transformacji
//    powrotnej
// - int N_s - liczba segmentow obrazu polarnego
// - int N_r - liczba pierscieni obrazu polarnego
// - float r_n - promien n-tego pierscienia = n*r_n n=0:N_r-1,
// - float blind - promien slepej plamki (ang. blind spot)
// - int x_0 - srodek kola (x)
// - int y_0 - srodek kola (y)
// - int src_width - szerokosc obrazu zrodlowego
// - int src_height - wysokosc obrazu zrodlowego
//Wartoœæ zwracana: void
//Wykorzystywane biblioteki: STL, OpenCV
void create_maps(cv::Mat & to_polar_map, cv::Mat & to_cart_map, int N_s, int N_r, float r_n, float blind, int x_0, int y_0, int src_width, int src_height);


//Autor: Piotr Konopka, Micha³ Góra (model matlab)
//Nazwa: get_polar_pixel
//Przeznaczenie: Funkcja wykorzystywana w create_maps() do utworzenia
// dla konkretnego piksela w obrazie polarnym
//Opis dzia³ania: Funkcja zapisuje w podanych mapach przekodowan przekodowania
// dla danego piksela obrazu polarnego.
//Argumenty:
// - int32_t * polar_coords - adres pozycji w mapie przekodowan dla danego
//    piksela w obrazie polarnym
// - int32_t * cart_coords - adres pozycji w mapie przekodowan dla danego
//    piksela w obrazie kartezjanskim powrotnym
// - int x_0 - srodek kola (x)
// - int y_0 - srodek kola (y)
// - float r_min - mniejszy promien wycinka kola
// - float r_max - wiekszy promien wycinka kola
// - float thet_min - mniejszy kat graniczny wycinku kola
// - float thet_max - wiekszy kat graniczny wycinku kola
// - int src_width - szerokosc obrazu zrodlowego
// - int N_s - liczba segmentow obrazu polarnego
// - int i - pozycja piksela obrazu polarnego ( 0 < i < N_r )
// - int j - pozycja piksela obrazu polarnego ( 0 < j < N_s )
//Wartoœæ zwracana: void
//Wykorzystywane biblioteki: STL
void get_polar_pixel(int32_t * polar_coords, int32_t * cart_coords, int x_0, int y_0, float r_min, float r_max, float thet_min, float thet_max, int src_width, int N_s, int i, int j );


#endif // POLAR_UTILS_HPP
